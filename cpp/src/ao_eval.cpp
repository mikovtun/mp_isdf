#include "ao_eval.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
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

LibcintWrapper make_tables(std::vector<FINT>& atm, std::vector<FINT>& bas,
                           std::vector<double>& env)
{
    if (atm.size() % ATM_SLOTS || bas.size() % BAS_SLOTS ||
        atm.size() / ATM_SLOTS > static_cast<std::size_t>(std::numeric_limits<FINT>::max()) ||
        bas.size() / BAS_SLOTS > static_cast<std::size_t>(std::numeric_limits<FINT>::max())) {
        throw std::invalid_argument("invalid libcint ATM/BAS layout");
    }
    const auto natm = static_cast<FINT>(atm.size() / ATM_SLOTS);
    const auto nbas = static_cast<FINT>(bas.size() / BAS_SLOTS);
    return LibcintWrapper(std::move(atm), std::move(bas), std::move(env), natm, nbas);
}

} // namespace

struct LibcintContext::Impl {
    using Device = Kokkos::DefaultExecutionSpace;
    Kokkos::View<Shell*, Device> d_shells;
    Kokkos::View<double*, Device> d_env;
    Kokkos::View<double*, Device> d_transforms;
    std::size_t nao = 0;
    std::size_t nshells = 0;
};

LibcintContext::LibcintContext(std::vector<FINT> atm, std::vector<FINT> bas,
                               std::vector<double> env)
    : tables_(make_tables(atm, bas, env)), impl_(std::make_unique<Impl>())
{
    if (!Kokkos::is_initialized()) {
        throw std::logic_error("Kokkos must be initialized before creating LibcintContext");
    }
    const auto& atoms = tables_.atm_vec();
    const auto& basis = tables_.bas_vec();
    const auto& environment = tables_.env_vec();
    const auto nbas = static_cast<std::size_t>(tables_.nbas());
    const auto natm = static_cast<std::size_t>(tables_.natm());
    std::vector<Shell> shells;
    shells.reserve(nbas);
    std::vector<double> transforms;
    std::size_t nao = 0;
    for (std::size_t i = 0; i < nbas; ++i) {
        const FINT* b = basis.data() + i * BAS_SLOTS;
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
        check_range(b[PTR_EXP], np, environment.size());
        check_range(b[PTR_COEFF], np * nc, environment.size());
        const FINT coord = atoms[static_cast<std::size_t>(atom) * ATM_SLOTS + PTR_COORD];
        check_range(coord, 3, environment.size());
        if (nao + static_cast<std::size_t>(nsph) * nc >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            transforms.size() + static_cast<std::size_t>(nsph) * ncart >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("AO or transformation index exceeds int range");
        }
        shells.push_back({l, np, nc, ncart, nsph,
                          b[PTR_EXP], b[PTR_COEFF], static_cast<int>(transforms.size()),
                          static_cast<int>(nao), environment[coord], environment[coord + 1], environment[coord + 2]});
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

    impl_->nao = nao;
    impl_->nshells = shells.size();
    impl_->d_shells = decltype(impl_->d_shells)("ao_shells", shells.size());
    impl_->d_env = decltype(impl_->d_env)("ao_env", environment.size());
    impl_->d_transforms = decltype(impl_->d_transforms)("ao_transform", transforms.size());
    auto h_shells = Kokkos::create_mirror_view(impl_->d_shells);
    auto h_env = Kokkos::create_mirror_view(impl_->d_env);
    auto h_transforms = Kokkos::create_mirror_view(impl_->d_transforms);
    for (std::size_t i = 0; i < shells.size(); ++i) h_shells(i) = shells[i];
    for (std::size_t i = 0; i < environment.size(); ++i) h_env(i) = environment[i];
    for (std::size_t i = 0; i < transforms.size(); ++i) h_transforms(i) = transforms[i];
    Kokkos::deep_copy(impl_->d_shells, h_shells);
    Kokkos::deep_copy(impl_->d_env, h_env);
    Kokkos::deep_copy(impl_->d_transforms, h_transforms);
}

LibcintContext::~LibcintContext() = default;

std::size_t LibcintContext::nao() const noexcept { return impl_->nao; }

template <class Device>
Kokkos::View<double*, Device> LibcintContext::evaluate_impl(
    const std::vector<std::array<double, 3>>& points, bool gradient) const
{
    if (!Kokkos::is_initialized()) {
        throw std::logic_error("Kokkos must be initialized before AO evaluation");
    }
    const std::size_t nao = impl_->nao;
    const std::size_t components = gradient ? 3 : 1;
    if (nao && (points.size() > std::numeric_limits<std::size_t>::max() / nao ||
                points.size() * nao > std::numeric_limits<std::size_t>::max() / components)) {
        throw std::overflow_error("AO result is too large");
    }
    Kokkos::View<double*, Device> d_result("ao_result", components * points.size() * nao);
    if (points.empty() || nao == 0) return d_result;

    Kokkos::View<Shell*, Device> d_shells;
    Kokkos::View<double*, Device> d_env, d_transforms;
    if constexpr (std::is_same_v<Device, Kokkos::DefaultExecutionSpace>) {
        d_shells = impl_->d_shells;
        d_env = impl_->d_env;
        d_transforms = impl_->d_transforms;
    } else {
        d_shells = decltype(d_shells)("ao_shells_on_host", impl_->d_shells.size());
        d_env = decltype(d_env)("ao_env_on_host", impl_->d_env.size());
        d_transforms = decltype(d_transforms)("ao_transforms_on_host", impl_->d_transforms.size());
        Kokkos::deep_copy(d_shells, impl_->d_shells);
        Kokkos::deep_copy(d_env, impl_->d_env);
        Kokkos::deep_copy(d_transforms, impl_->d_transforms);
    }
    Kokkos::View<double*[3], Device> d_points("ao_points", points.size());
    auto h_points = Kokkos::create_mirror_view(d_points);
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (int k = 0; k < 3; ++k) h_points(i, k) = points[i][k];
    }
    Kokkos::deep_copy(d_points, h_points);

    // One work item per (grid point, shell). Every item owns disjoint output
    // columns; no atomics or device-side allocation are needed.
    using Policy = Kokkos::MDRangePolicy<Device, Kokkos::Rank<2>>;
    const std::size_t npoints = points.size();
    Kokkos::parallel_for("isdf_eval_ao",
                         Policy({0, 0}, {npoints, impl_->nshells}),
                         KOKKOS_LAMBDA(const std::size_t p, const std::size_t sh) {
        const Shell b = d_shells(sh);
        const double x = d_points(p, 0) - b.x;
        const double y = d_points(p, 1) - b.y;
        const double z = d_points(p, 2) - b.z;
        const double r2 = x*x + y*y + z*z;
        for (int ctr = 0; ctr < b.nctr; ++ctr) {
            double radial = 0.0;
            double radial_prime = 0.0; // derivative w.r.t. r^2
            for (int prim = 0; prim < b.nprim; ++prim) {
                const double alpha = d_env(b.exp_offset + prim);
                const double term = d_env(b.coeff_offset + ctr * b.nprim + prim) *
                                    exp(-alpha * r2);
                radial += term;
                if (gradient) radial_prime -= alpha * term;
            }
            for (int s = 0; s < b.nsph; ++s) {
                double value = 0.0, dx = 0.0, dy = 0.0, dz = 0.0;
                for (int c = 0; c < b.ncart; ++c) {
                    int lx, ly, lz;
                    cartesian_powers(b.l, c, lx, ly, lz);
                    const double px = integer_power(x, lx);
                    const double py = integer_power(y, ly);
                    const double pz = integer_power(z, lz);
                    const double t = d_transforms(b.transform_offset + s*b.ncart + c);
                    value += t * px * py * pz;
                    if (gradient) {
                        if (lx) dx += t * lx * integer_power(x, lx-1) * py * pz;
                        if (ly) dy += t * ly * px * integer_power(y, ly-1) * pz;
                        if (lz) dz += t * lz * px * py * integer_power(z, lz-1);
                    }
                }
                const std::size_t index = p * nao + b.ao_offset + ctr * b.nsph + s;
                if (gradient) {
                    const std::size_t stride = npoints * nao;
                    d_result(index) = radial * dx + 2.0 * x * radial_prime * value;
                    d_result(stride + index) = radial * dy + 2.0 * y * radial_prime * value;
                    d_result(2 * stride + index) = radial * dz + 2.0 * z * radial_prime * value;
                } else {
                    d_result(index) = radial * value;
                }
            }
        }
    });
    return d_result;
}

Kokkos::View<double*, Kokkos::DefaultExecutionSpace> LibcintContext::evaluate_default_view(
    const std::vector<std::array<double, 3>>& points, bool gradient) const
{
    return evaluate_impl<Kokkos::DefaultExecutionSpace>(points, gradient);
}

Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> LibcintContext::evaluate_host_view(
    const std::vector<std::array<double, 3>>& points, bool gradient) const
{
    return evaluate_impl<Kokkos::DefaultHostExecutionSpace>(points, gradient);
}

#ifdef KOKKOS_ENABLE_CUDA
Kokkos::View<double*, Kokkos::Cuda> LibcintContext::evaluate_cuda_view(
    const std::vector<std::array<double, 3>>& points, bool gradient) const
{
    return evaluate_impl<Kokkos::Cuda>(points, gradient);
}
#endif

std::vector<double> LibcintContext::evaluate(
    const std::vector<std::array<double, 3>>& points, bool gradient,
    EvaluationSpace space) const
{
    auto to_host = [](const auto& values) {
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
        std::vector<double> result(values.size());
        for (std::size_t i = 0; i < result.size(); ++i) result[i] = mirror(i);
        return result;
    };
    switch (space) {
    case EvaluationSpace::Host:
        return to_host(evaluate_host_view(points, gradient));
    case EvaluationSpace::Auto:
#ifdef KOKKOS_ENABLE_CUDA
    case EvaluationSpace::Device:
        return to_host(evaluate_cuda_view(points, gradient));
#else
        return to_host(evaluate_host_view(points, gradient));
    case EvaluationSpace::Device:
        throw std::invalid_argument("device evaluation requires a CUDA build");
#endif
    }
    throw std::invalid_argument("unknown evaluation space");
}

std::vector<double> LibcintContext::eval_ao(
    const std::vector<std::array<double, 3>>& points, EvaluationSpace space) const
{
    return evaluate(points, false, space);
}

std::vector<double> LibcintContext::eval_ao_gradient(
    const std::vector<std::array<double, 3>>& points, EvaluationSpace space) const
{
    return evaluate(points, true, space);
}

std::vector<double> eval_ao(const std::vector<FINT>& atm,
                            const std::vector<FINT>& bas,
                            const std::vector<double>& env,
                            const std::vector<std::array<double, 3>>& points,
                            EvaluationSpace space)
{
    return LibcintContext(atm, bas, env).eval_ao(points, space);
}

std::vector<double> eval_ao_gradient(
    const std::vector<FINT>& atm,
    const std::vector<FINT>& bas,
    const std::vector<double>& env,
    const std::vector<std::array<double, 3>>& points,
    EvaluationSpace space)
{
    return LibcintContext(atm, bas, env).eval_ao_gradient(points, space);
}

} // namespace isdf
