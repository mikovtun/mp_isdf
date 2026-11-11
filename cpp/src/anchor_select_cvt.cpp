#include "anchor_select_cvt.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {

template<class Space>
std::vector<std::size_t> run(const std::vector<std::array<double, 3>>& points,
                             const std::vector<std::size_t>& active,
                             const std::vector<double>& mass, std::size_t rank,
                             std::uint64_t seed, std::size_t max_iter, double tol, bool debug)
{
    const std::size_t n = active.size();
    Kokkos::View<double*, Space> xyz("cvt_points", 3*n), w("cvt_mass", n);
    Kokkos::View<double*, Space> centers("cvt_centers", 3*rank), nearest("cvt_nearest", n);
    Kokkos::View<double*, Space> sums("cvt_sums", 4*rank);
    Kokkos::View<double*, Space> residual("cvt_residual", n);
    auto hx = Kokkos::create_mirror_view(xyz);
    auto hw = Kokkos::create_mirror_view(w);
    for (std::size_t i = 0; i < n; ++i) {
        hw(i) = mass[i];
        for (int d = 0; d < 3; ++d) hx(3*i+d) = points[active[i]][d];
    }
    Kokkos::deep_copy(xyz, hx);
    Kokkos::deep_copy(w, hw);
    auto hc = Kokkos::create_mirror_view(centers);
    std::vector<std::array<double, 3>> c(rank);
    std::mt19937_64 rng(seed);
    // Sample from unnormalized masses, avoiding probability normalization overflow.
    auto sample = [&](const std::vector<double>& scores) {
        long double total = 0;
        for (double v : scores) total += v;
        if (!(total > 0)) throw std::invalid_argument("CVT importance has zero total mass");
        std::uniform_real_distribution<long double> draw(0, total);
        const long double target = draw(rng);
        long double prefix = 0;
        std::size_t last = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (scores[i] <= 0) continue;
            prefix += scores[i];
            last = i;
            if (target < prefix) return i;
        }
        return last;
    };
    const auto first = sample(mass);
    c[0] = points[active[first]];
    std::vector<double> scores(n);
    for (std::size_t k = 1; k < rank; ++k) {
        const double px = c[k-1][0], py = c[k-1][1], pz = c[k-1][2];
        Kokkos::parallel_for("cvt_seed_distance", Kokkos::RangePolicy<Space>(0, n),
            KOKKOS_LAMBDA(std::size_t i) {
                double d2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const double delta = xyz(3*i+d) - (d == 0 ? px : d == 1 ? py : pz);
                    d2 += delta*delta;
                }
                nearest(i) = k == 1 ? d2 : (d2 < nearest(i) ? d2 : nearest(i));
            });
        auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nearest);
        long double total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            scores[i] = mass[i] * hn(i);
            if (!std::isfinite(scores[i])) throw std::invalid_argument("CVT weighted distance overflow");
            total += scores[i];
        }
        // Coincident sites: fall back to mass sampling, as in the Python reference.
        c[k] = points[active[sample(total > 0 ? scores : mass)]];
    }
    std::array<double, 3> lo = points[active[0]], hi = lo;
    for (auto i : active) for (int d = 0; d < 3; ++d) {
        lo[d] = std::min(lo[d], points[i][d]);
        hi[d] = std::max(hi[d], points[i][d]);
    }
    double scale2 = 0;
    for (int d = 0; d < 3; ++d) scale2 += (hi[d]-lo[d])*(hi[d]-lo[d]);
    const double scale = std::max(std::sqrt(scale2), 1.0);
    std::size_t iteration = 0;
    for (; iteration < max_iter; ++iteration) {
        for (std::size_t k = 0; k < rank; ++k)
            for (int d = 0; d < 3; ++d) hc(3*k+d) = c[k][d];
        Kokkos::deep_copy(centers, hc);
        Kokkos::deep_copy(sums, 0.0);
        Kokkos::parallel_for("cvt_lloyd", Kokkos::RangePolicy<Space>(0, n),
            KOKKOS_LAMBDA(std::size_t i) {
                double best = 1.7976931348623157e308;
                std::size_t label = 0;
                for (std::size_t k = 0; k < rank; ++k) {
                    double d2 = 0;
                    for (int d = 0; d < 3; ++d) {
                        const double delta = xyz(3*i+d) - centers(3*k+d);
                        d2 += delta*delta;
                    }
                    if (d2 < best) { best = d2; label = k; }
                }
                const double m = w(i);
                Kokkos::atomic_add(&sums(4*label), m);
                for (int d = 0; d < 3; ++d)
                    Kokkos::atomic_add(&sums(4*label+1+d), m*xyz(3*i+d));
                residual(i) = m*best;
            });
        auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sums);
        auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, residual);
        std::vector<std::array<double, 3>> next(rank);
        std::vector<std::size_t> empty;
        for (std::size_t k = 0; k < rank; ++k) {
            if (hs(4*k) > 0) {
                for (int d = 0; d < 3; ++d) next[k][d] = hs(4*k+1+d)/hs(4*k);
            } else empty.push_back(k);
        }
        std::vector<unsigned char> reseeded(n, 0);
        for (auto k : empty) {
            std::size_t site = n;
            double best = -1;
            for (std::size_t i = 0; i < n; ++i)
                if (!reseeded[i] && hr(i) > best) { best = hr(i); site = i; }
            reseeded[site] = 1;
            next[k] = points[active[site]];
        }
        double shift2 = 0;
        for (std::size_t k = 0; k < rank; ++k)
            for (int d = 0; d < 3; ++d) {
                const double delta = next[k][d] - c[k][d];
                shift2 += delta*delta;
            }
        c.swap(next);
        if (std::sqrt(shift2) <= tol*scale) { ++iteration; break; }
    }
    // Snapping uses only positive-importance points; resolve equal distances
    // by original grid index and never select the same site twice.
    std::vector<unsigned char> used(n, 0);
    std::vector<std::size_t> indices;
    indices.reserve(rank);
    for (const auto& center : c) {
        std::size_t site = n;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            if (used[i]) continue;
            double d2 = 0;
            for (int d = 0; d < 3; ++d) {
                const double delta = points[active[i]][d] - center[d];
                d2 += delta*delta;
            }
            if (d2 < best || (d2 == best && (site == n || active[i] < active[site]))) {
                best = d2; site = i;
            }
        }
        used[site] = 1;
        indices.push_back(active[site]);
    }
    std::sort(indices.begin(), indices.end());
    if (debug) std::cerr << "[CVT] Lloyd iterations: " << iteration << ", anchors: " << rank << '\n';
    return indices;
}
} // namespace

std::vector<std::size_t> CVTAnchorSelection::select(
    const std::vector<std::array<double, 3>>& points, const std::vector<double>& weights,
    const std::vector<double>& importance, std::size_t rank, EvaluationSpace space,
    std::uint64_t seed, std::size_t max_iter, double tol, bool debug)
{
    if (points.size() != weights.size() || points.size() != importance.size())
        throw std::invalid_argument("CVT points, quadrature weights and CVT weights must have equal lengths");
    if (rank == 0 || max_iter == 0 || !std::isfinite(tol) || tol < 0)
        throw std::invalid_argument("CVT requires positive rank and max_iter, finite nonnegative tol");
    std::vector<std::size_t> active;
    std::vector<double> mass;
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (double x : points[i]) if (!std::isfinite(x))
            throw std::invalid_argument("CVT coordinates must be finite");
        if (!std::isfinite(weights[i]) || !std::isfinite(importance[i]) || importance[i] < 0)
            throw std::invalid_argument("CVT weights must be finite and nonnegative");
        // Quadrature weights gate eligible grid sites, but do not multiply
        // electron-density importance. The two CVT weighting modes are exclusive.
        const double m = weights[i] == 0 ? 0.0 : importance[i];
        if (!std::isfinite(m)) throw std::invalid_argument("CVT importance overflow");
        if (m > 0) { active.push_back(i); mass.push_back(m); }
    }
    if (rank > active.size()) throw std::invalid_argument("CVT requires at least rank positive-importance sites");
    switch (space) {
    case EvaluationSpace::Host:
        return run<Kokkos::DefaultHostExecutionSpace>(points, active, mass, rank, seed, max_iter, tol, debug);
    case EvaluationSpace::Auto:
#ifdef KOKKOS_ENABLE_CUDA
    case EvaluationSpace::Device:
        return run<Kokkos::Cuda>(points, active, mass, rank, seed, max_iter, tol, debug);
#else
        return run<Kokkos::DefaultHostExecutionSpace>(points, active, mass, rank, seed, max_iter, tol, debug);
    case EvaluationSpace::Device:
        throw std::invalid_argument("device evaluation requires a CUDA build");
#endif
    }
    throw std::invalid_argument("unknown evaluation space");
}
} // namespace isdf
