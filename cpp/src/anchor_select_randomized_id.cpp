#include "anchor_select_randomized_id.hpp"
#include "anchor_select_qrcp_backend.hpp"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace isdf {
namespace {

// Small-case independent audit: actual packed pair matrix and equally ranked
// *orthogonal projection* errors, not sketch or ID interpolation errors.
void audit_projection(const AnchorSelection& selector,
                      const std::vector<std::array<double, 3>>& points,
                      const std::vector<double>& weights,
                      const std::vector<AxisType>& axes,
                      const std::vector<double>& coeff, std::size_t nocc,
                      const std::vector<std::size_t>& selected)
{
    const std::size_t n = points.size();
    const auto axis = selector.evaluate_axis<Kokkos::DefaultHostExecutionSpace>(
        points, axes, coeff, nocc);
    const auto features = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, axis.values);
    const std::size_t d = axis.width, m = d*(d+1)/2;
    std::vector<double> matrix(m*n);
    for (std::size_t g = 0; g < n; ++g) {
        const double w = std::sqrt(std::abs(weights[g]));
        std::size_t row = 0;
        for (std::size_t mu = 0; mu < d; ++mu)
            for (std::size_t nu = mu; nu < d; ++nu)
                matrix[g*m+row++] = w*features(g*d+mu)*features(g*d+nu)*
                    (mu == nu ? 1.0 : std::sqrt(2.0));
    }
    std::vector<double> dense(matrix);
    const auto baseline = detail::qrcp_cpu(dense.data(), m, n);
    auto error = [&](const std::vector<std::size_t>& pivots) {
        std::vector<double> q;
        q.reserve(m*pivots.size());
        for (const auto g : pivots) {
            std::vector<double> v(matrix.begin()+g*m, matrix.begin()+(g+1)*m);
            double original_norm = 0;
            for (double x : v) original_norm = std::hypot(original_norm, x);
            for (int pass = 0; pass < 2; ++pass)
                for (std::size_t j = 0; j < q.size()/m; ++j) {
                    double dot = 0;
                    for (std::size_t i = 0; i < m; ++i) dot += q[j*m+i]*v[i];
                    for (std::size_t i = 0; i < m; ++i) v[i] -= dot*q[j*m+i];
                }
            double norm = 0;
            for (double x : v) norm = std::hypot(norm, x);
            if (norm <= 32*std::numeric_limits<double>::epsilon()*original_norm) continue;
            for (double x : v) q.push_back(x/norm);
        }
        double total = 0, residual = 0;
        for (std::size_t g = 0; g < n; ++g) {
            for (std::size_t i = 0; i < m; ++i) total += matrix[g*m+i]*matrix[g*m+i];
            std::vector<double> v(matrix.begin()+g*m, matrix.begin()+(g+1)*m);
            for (int pass = 0; pass < 2; ++pass)
                for (std::size_t j = 0; j < q.size()/m; ++j) {
                    double dot = 0;
                    for (std::size_t i = 0; i < m; ++i) dot += q[j*m+i]*v[i];
                    for (std::size_t i = 0; i < m; ++i) v[i] -= dot*q[j*m+i];
                }
            for (double x : v) residual += x*x;
        }
        return total ? std::sqrt(residual/total) : 0.0;
    };
    std::vector<std::size_t> dense_pivots;
    for (std::size_t j = 0; j < selected.size(); ++j)
        dense_pivots.push_back(static_cast<std::size_t>(baseline[j]-1));
    std::cerr << "[randomized id audit] rank=" << selected.size()
              << " actual_projection_relative_error=" << error(selected)
              << " dense_qrcp_projection_relative_error=" << error(dense_pivots)
              << " (equal rank; not ID interpolation error)\n";
}

} // namespace

std::vector<std::size_t> RandomizedIDAnchorSelection::select(
    const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights, const std::vector<AxisType>& axes,
    double tolerance, std::size_t max_rank,
    const std::vector<double>& mo_coeff, std::size_t nocc,
    EvaluationSpace space, std::size_t batch_size, bool debug,
    std::size_t sketch_size, std::size_t oversampling,
    std::uint64_t seed, bool audit)
{
    if (points.size() != weights.size() || !std::isfinite(tolerance) || tolerance < 0 ||
        batch_size == 0 || oversampling == 0 || sketch_size <= oversampling ||
        (max_rank && max_rank > sketch_size-oversampling))
        throw std::invalid_argument("invalid randomized ID grid, tolerance, batch, sketch or rank/oversampling");
    const auto spec = selector.axis_spec(axes, mo_coeff, nocc);
    if (spec.width > std::numeric_limits<std::size_t>::max()/spec.width ||
        spec.width+1 > std::numeric_limits<std::size_t>::max()/spec.width ||
        sketch_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        points.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("randomized ID dimension exceeds supported range");
    std::vector<std::array<double, 3>> kept_points;
    std::vector<double> kept_weights;
    std::vector<std::size_t> original;
    kept_points.reserve(points.size());
    kept_weights.reserve(points.size());
    original.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!std::isfinite(weights[i])) throw std::invalid_argument("grid weights must be finite");
        if (weights[i] == 0) continue;
        for (double x : points[i])
            if (!std::isfinite(x)) throw std::invalid_argument("grid coordinates must be finite");
        kept_points.push_back(points[i]);
        kept_weights.push_back(weights[i]);
        original.push_back(i);
    }
    if (space == EvaluationSpace::Device) {
#ifndef ISDF_ENABLE_CUDA
        throw std::invalid_argument("device evaluation requires a CUDA build");
#endif
    } else if (space != EvaluationSpace::Auto && space != EvaluationSpace::Host) {
        throw std::invalid_argument("unknown evaluation space");
    }
    const auto n = original.size(), d = spec.width;
    if (audit && (d*(d+1)/2 > 2000000/std::max<std::size_t>(1,n)))
        throw std::invalid_argument("randomized ID dense audit is limited to 2 million pair-matrix elements");
    if (n == 0) return {};
    if (sketch_size > std::numeric_limits<std::size_t>::max()/n ||
        sketch_size > std::numeric_limits<std::size_t>::max()/std::min(batch_size,n))
        throw std::overflow_error("randomized ID sketch is too large");
    if (space == EvaluationSpace::Host)
        return select_in_space<Kokkos::DefaultHostExecutionSpace>(selector, kept_points,
            kept_weights, original, axes, tolerance, max_rank, mo_coeff, nocc,
            batch_size, debug, sketch_size, oversampling, seed, audit);
#ifdef ISDF_ENABLE_CUDA
    return select_in_space<Kokkos::Cuda>(selector, kept_points, kept_weights,
        original, axes, tolerance, max_rank, mo_coeff, nocc,
        batch_size, debug, sketch_size, oversampling, seed, audit);
#else
    return select_in_space<Kokkos::DefaultHostExecutionSpace>(selector, kept_points,
        kept_weights, original, axes, tolerance, max_rank, mo_coeff, nocc,
        batch_size, debug, sketch_size, oversampling, seed, audit);
#endif
}

template <class Space>
std::vector<std::size_t> RandomizedIDAnchorSelection::select_in_space(
    const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights, const std::vector<std::size_t>& original,
    const std::vector<AxisType>& axes, double tolerance, std::size_t max_rank,
    const std::vector<double>& mo_coeff, std::size_t nocc,
    std::size_t batch_size, bool debug, std::size_t ell, std::size_t oversampling,
    std::uint64_t seed, bool audit)
{
    const auto started = std::chrono::steady_clock::now();
    const std::size_t d = selector.axis_spec(axes, mo_coeff, nocc).width;
    const std::size_t n = original.size();
    Kokkos::View<double*, Space> u("id_random_u", d*ell), v("id_random_v", d*ell);
    auto hu = Kokkos::create_mirror_view(u), hv = Kokkos::create_mirror_view(v);
    // Stable across batch sizes; two independent streams of random factors.
    std::mt19937_64 gen_u(seed), gen_v(seed ^ 0x9e3779b97f4a7c15ULL);
    std::normal_distribution<double> normal_u(0,1), normal_v(0,1);
    for (std::size_t i = 0; i < d*ell; ++i) {
        hu(i) = normal_u(gen_u);
        hv(i) = normal_v(gen_v);
    }
    Kokkos::deep_copy(u, hu);
    Kokkos::deep_copy(v, hv);
    Kokkos::View<double*, Space> y("id_sketch", ell*n); // column-major Y[s,g]
    for (std::size_t start = 0; start < n; start += std::min(batch_size,n-start)) {
        const auto count = std::min(batch_size,n-start);
        std::vector<std::array<double,3>> batch(points.begin()+start, points.begin()+start+count);
        auto axis = selector.evaluate_axis<Space>(batch, axes, mo_coeff, nocc);
        auto phi = axis.values;
        Kokkos::View<double*, Space> sw("id_weights", count);
        auto hw = Kokkos::create_mirror_view(sw);
        for (std::size_t g = 0; g < count; ++g) hw(g) = std::sqrt(std::abs(weights[start+g]));
        Kokkos::deep_copy(sw, hw);
        const double scaling = 1/std::sqrt(static_cast<double>(ell));
        Kokkos::parallel_for("id_factored_sketch", Kokkos::RangePolicy<Space>(0,count*ell),
            KOKKOS_LAMBDA(const std::size_t i) {
                const std::size_t g = i/ell, s = i%ell;
                double left = 0, right = 0;
                for (std::size_t a = 0; a < d; ++a) {
                    const double value = phi(g*d+a);
                    left += value*u(a*ell+s);
                    right += value*v(a*ell+s);
                }
                y((start+g)*ell+s) = scaling*sw(g)*left*right;
            });
        int bad = 0;
        Kokkos::parallel_reduce("id_check_sketch", Kokkos::RangePolicy<Space>(start*ell,(start+count)*ell),
            KOKKOS_LAMBDA(const std::size_t i, int& invalid) {
                if (!Kokkos::isfinite(y(i))) ++invalid;
            }, bad);
        if (bad) throw std::overflow_error("non-finite randomized ID sketch");
    }
    Kokkos::fence();
    std::vector<std::int64_t> jpvt;
#ifdef ISDF_ENABLE_CUDA
    if constexpr (std::is_same_v<Space, Kokkos::Cuda>)
        jpvt = detail::qrcp_magma(y.data(), ell, n);
    else
#endif
        jpvt = detail::qrcp_cpu(y.data(), ell, n);
    const std::size_t steps = std::min(ell,n);
    Kokkos::View<double*, Space> energy("id_qr_row_energy", steps);
    Kokkos::View<double*, Space> diag("id_qr_diagonal", steps);
    Kokkos::parallel_for("id_qr_residual_rows", Kokkos::RangePolicy<Space>(0,steps),
        KOKKOS_LAMBDA(const std::size_t i) {
            double sum = 0;
            for (std::size_t g = i; g < n; ++g) {
                const double r = y(g*ell+i); // below diagonal stores reflectors
                sum += r*r;
            }
            energy(i) = sum;
            diag(i) = Kokkos::abs(y(i*ell+i));
        });
    auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, energy);
    auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, diag);
    std::vector<double> tails(steps+1, 0);
    double diag_max = 0;
    for (std::size_t i = steps; i-- > 0;) {
        tails[i] = std::hypot(tails[i+1], std::sqrt(he(i)));
        diag_max = std::max(diag_max, hd(i));
    }
    if (!std::isfinite(tails[0]) || !std::isfinite(diag_max))
        throw std::overflow_error("non-finite randomized ID QR residual");
    const double norm = tails[0];
    const double numeric_floor = std::numeric_limits<double>::epsilon()*std::max(ell,n)*diag_max;
    std::size_t numerical_rank = steps;
    for (std::size_t i = 0; i < steps; ++i)
        if (hd(i) <= numeric_floor) { numerical_rank = i; break; }
    const std::size_t physical_rank = d*(d+1)/2;
    const std::size_t cap = std::min({max_rank ? max_rank : ell-oversampling,
                                      ell-oversampling, steps, physical_rank, numerical_rank});
    std::size_t rank = cap;
    for (std::size_t i = 0; i <= cap; ++i) {
        if (tails[i] <= tolerance*norm) { rank = i; break; }
    }
    std::vector<std::size_t> selected;
    selected.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        if (jpvt[i] < 1 || static_cast<std::size_t>(jpvt[i]) > n)
            throw std::runtime_error("randomized ID QRCP returned invalid pivot");
        selected.push_back(original[static_cast<std::size_t>(jpvt[i]-1)]);
    }
    if (debug) {
        std::cerr << "[randomized id] rank=" << rank << " sketch_size=" << ell
                  << " oversampling=" << oversampling << " columns=" << n
                  << " sketch_relative_frobenius=" << (norm ? tails[rank]/norm : 0)
                  << " tolerance_met=" << (tails[rank] <= tolerance*norm)
                  << " numerical_rank=" << numerical_rank
                  << " rank_cap=" << cap << " seed=" << seed
                  << " space=" << (std::is_same_v<Space,Kokkos::DefaultHostExecutionSpace> ? "host" : "device")
                  << " elapsed=" << std::chrono::duration<double>(
                      std::chrono::steady_clock::now()-started).count() << " s\n";
    }
    if (audit) {
        // Convert original point IDs to positions within the retained grid.
        std::vector<std::size_t> local;
        local.reserve(selected.size());
        for (auto id : selected)
            local.push_back(static_cast<std::size_t>(
                std::lower_bound(original.begin(), original.end(), id)-original.begin()));
        audit_projection(selector, points, weights, axes, mo_coeff, nocc, local);
    }
    // Future full-ID continuation: retain y's packed QR, jpvt, and QR tau in a
    // result object, then solve R11 * X = R[:rank,:] in grid batches. No A^T A.
    return selected;
}

} // namespace isdf
