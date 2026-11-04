#include "ao_eval.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {

struct Shell {
    int l, nprim, nctr, ncart, nsph;
    int exp_offset, coeff_offset, transform_offset, ao_offset;
    double x, y, z;
};

KOKKOS_INLINE_FUNCTION
double integer_power(double x, int n)
{
    double value = 1.0;
    for (int i = 0; i < n; ++i) value *= x;
    return value;
}

// The order is x^l, x^(l-1)y, x^(l-1)z, ... matching libcint.
KOKKOS_INLINE_FUNCTION
void cartesian_powers(int l, int index, int& lx, int& ly, int& lz)
{
    for (lx = l; lx >= 0; --lx) {
        const int count = l - lx + 1;
        if (index < count) {
            ly = l - lx - index;
            lz = index;
            return;
        }
        index -= count;
    }
    lx = ly = lz = 0;
}

void check_range(FINT offset, FINT length, std::size_t size)
{
    if (offset < 0 || length < 0 || static_cast<std::size_t>(offset) > size ||
        static_cast<std::size_t>(length) > size - static_cast<std::size_t>(offset)) {
        throw std::invalid_argument("libcint table contains an invalid ENV offset");
    }
}

} // namespace

std::vector<double> eval_ao(const std::vector<FINT>& atm,
                            const std::vector<FINT>& bas,
                            const std::vector<double>& env,
                            const std::vector<std::array<double, 3>>& points)
{
    if (!Kokkos::is_initialized()) {
        throw std::logic_error("Kokkos must be initialized before eval_ao");
    }
    if (atm.size() % ATM_SLOTS || bas.size() % BAS_SLOTS || env.size() < PTR_ENV_START) {
        throw std::invalid_argument("invalid libcint ATM/BAS/ENV layout");
    }

    const auto nbas = bas.size() / BAS_SLOTS;
    const auto natm = atm.size() / ATM_SLOTS;
    std::vector<Shell> shells;
    shells.reserve(nbas);
    std::vector<double> transforms;
    std::size_t nao = 0;
    for (std::size_t i = 0; i < nbas; ++i) {
        const FINT* b = bas.data() + i * BAS_SLOTS;
        const FINT atom = b[ATOM_OF];
        const FINT l = b[ANG_OF];
        const FINT np = b[NPRIM_OF];
        const FINT nc = b[NCTR_OF];
        if (atom < 0 || static_cast<std::size_t>(atom) >= natm ||
            l < 0 || l > ANG_MAX || np <= 0 || nc <= 0 ||
            np > NPRIM_MAX || nc > NCTR_MAX) {
            throw std::invalid_argument("invalid libcint basis shell");
        }
        const int ncart = (l + 1) * (l + 2) / 2;
        const int nsph = 2 * l + 1;
        check_range(b[PTR_EXP], np, env.size());
        check_range(b[PTR_COEFF], np * nc, env.size());
        const FINT coord = atm[static_cast<std::size_t>(atom) * ATM_SLOTS + PTR_COORD];
        check_range(coord, 3, env.size());
        if (nao + static_cast<std::size_t>(nsph) * nc >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            transforms.size() + static_cast<std::size_t>(nsph) * ncart >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("AO or transformation index exceeds int range");
        }
        shells.push_back({l, np, nc, ncart, nsph,
                          b[PTR_EXP], b[PTR_COEFF], static_cast<int>(transforms.size()),
                          static_cast<int>(nao), env[coord], env[coord + 1], env[coord + 2]});
        nao += static_cast<std::size_t>(nsph) * nc;

        // Generate libcint's exact real-spherical cart-to-spherical matrix on
        // the host. Its s/p kernels return the input pointer unchanged.
        std::vector<double> cart(ncart, 0.0), sph(nsph, 0.0);
        for (int c = 0; c < ncart; ++c) {
            cart[c] = 1.0;
            const double* result = CINTc2s_bra_sph(sph.data(), 1, cart.data(), l);
            if (!result) throw std::runtime_error("libcint spherical transform failed");
            if (c == 0) transforms.resize(transforms.size() + nsph * ncart);
            const auto base = shells.back().transform_offset;
            for (int s = 0; s < nsph; ++s) {
                // libcint's s/p integral fast paths return the Cartesian
                // input unchanged; AO evaluation still needs normalized
                // real spherical harmonics for those angular momenta.
                const double scale = l == 0 ? 1.0 / std::sqrt(4.0 * std::numbers::pi)
                                   : l == 1 ? std::sqrt(3.0 / (4.0 * std::numbers::pi))
                                            : 1.0;
                transforms[base + s * ncart + c] = scale * result[s];
            }
            cart[c] = 0.0;
        }
    }

    if (nao && points.size() > std::numeric_limits<std::size_t>::max() / nao) {
        throw std::overflow_error("AO result is too large");
    }
    std::vector<double> result(points.size() * nao);
    if (points.empty() || nao == 0) return result;

    using Device = Kokkos::DefaultExecutionSpace;
    Kokkos::View<Shell*, Device> d_shells("ao_shells", shells.size());
    Kokkos::View<double*, Device> d_env("ao_env", env.size());
    Kokkos::View<double*, Device> d_transforms("ao_transform", transforms.size());
    Kokkos::View<double*[3], Device> d_points("ao_points", points.size());
    Kokkos::View<double**, Kokkos::LayoutRight, Device> d_result("ao_result", points.size(), nao);
    auto h_shells = Kokkos::create_mirror_view(d_shells);
    auto h_env = Kokkos::create_mirror_view(d_env);
    auto h_transforms = Kokkos::create_mirror_view(d_transforms);
    auto h_points = Kokkos::create_mirror_view(d_points);
    for (std::size_t i = 0; i < shells.size(); ++i) h_shells(i) = shells[i];
    for (std::size_t i = 0; i < env.size(); ++i) h_env(i) = env[i];
    for (std::size_t i = 0; i < transforms.size(); ++i) h_transforms(i) = transforms[i];
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (int k = 0; k < 3; ++k) h_points(i, k) = points[i][k];
    }
    Kokkos::deep_copy(d_shells, h_shells);
    Kokkos::deep_copy(d_env, h_env);
    Kokkos::deep_copy(d_transforms, h_transforms);
    Kokkos::deep_copy(d_points, h_points);

    // One work item per (grid point, shell). Every item owns disjoint output
    // columns; no atomics or device-side allocation are needed.
    using Policy = Kokkos::MDRangePolicy<Device, Kokkos::Rank<2>>;
    Kokkos::parallel_for("isdf_eval_ao",
                         Policy({0, 0}, {points.size(), shells.size()}),
                         KOKKOS_LAMBDA(const std::size_t p, const std::size_t sh) {
        const Shell b = d_shells(sh);
        const double x = d_points(p, 0) - b.x;
        const double y = d_points(p, 1) - b.y;
        const double z = d_points(p, 2) - b.z;
        const double r2 = x*x + y*y + z*z;
        for (int ctr = 0; ctr < b.nctr; ++ctr) {
            double radial = 0.0;
            for (int prim = 0; prim < b.nprim; ++prim) {
                radial += d_env(b.coeff_offset + ctr * b.nprim + prim) *
                          exp(-d_env(b.exp_offset + prim) * r2);
            }
            for (int s = 0; s < b.nsph; ++s) {
                double value = 0.0;
                for (int c = 0; c < b.ncart; ++c) {
                    int lx, ly, lz;
                    cartesian_powers(b.l, c, lx, ly, lz);
                    value += d_transforms(b.transform_offset + s*b.ncart + c) *
                             integer_power(x, lx) * integer_power(y, ly) *
                             integer_power(z, lz);
                }
                d_result(p, b.ao_offset + ctr * b.nsph + s) = radial * value;
            }
        }
    });
    auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_result);
    for (std::size_t p = 0; p < points.size(); ++p) {
        for (std::size_t a = 0; a < nao; ++a) result[p * nao + a] = h_result(p, a);
    }
    return result;
}

} // namespace isdf
