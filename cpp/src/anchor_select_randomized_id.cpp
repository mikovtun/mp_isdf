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
                      const std::vector<std::size_t>& selected,
                      const std::vector<double>& axis_scales,
                      const std::vector<double>& importance = {},
                      const std::vector<double>& ao_norms = {},
                      const char* objective = "physical_quadrature")
{
    const std::size_t n = points.size();
    const auto axis = selector.evaluate_axis<Kokkos::DefaultHostExecutionSpace>(
        points, axes, coeff, nocc, axis_scales);
    const auto features = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, axis.values);
    const std::size_t d = axis.width, m = d*(d+1)/2;
    std::vector<double> matrix(m*n);
    for (std::size_t g = 0; g < n; ++g) {
        const double w = std::sqrt(importance.empty() ? std::abs(weights[g]) : importance[g]);
        const double inverse_norm = ao_norms.empty() ? 1.0 : 1/std::sqrt(ao_norms[g]);
        std::size_t row = 0;
        for (std::size_t mu = 0; mu < d; ++mu)
            for (std::size_t nu = mu; nu < d; ++nu)
                matrix[g*m+row++] = w*(features(g*d+mu)*inverse_norm)*
                    (features(g*d+nu)*inverse_norm)*(mu == nu ? 1.0 : std::sqrt(2.0));
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
    std::cerr << "[randomized id audit] objective=" << objective << " rank=" << selected.size()
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
    std::uint64_t seed, bool audit, SketchPolicy policy,
    const std::vector<double>& axis_scales,
    const std::vector<double>& point_importance, double ao_screening_threshold)
{
    const bool density = policy == SketchPolicy::AODensity || policy == SketchPolicy::AODensityNormalized;
    if (density) {
        if (point_importance.size() != points.size())
            throw std::invalid_argument("point_importance must have shape (ngrid,)");
        for (double m : point_importance)
            if (!std::isfinite(m) || m < 0)
                throw std::invalid_argument("point_importance must be finite and nonnegative");
        if (!std::isfinite(ao_screening_threshold) || ao_screening_threshold < 0)
            throw std::invalid_argument("AO screening threshold must be finite and nonnegative");
    }
    if (density && (axes.size() != 1 || axes[0] != AxisType::AtomicOrbital))
        throw std::invalid_argument("density Khatri-Rao sketches with non-AO axis features are NYI");
    if (points.size() != weights.size() || !std::isfinite(tolerance) || tolerance < 0 ||
        batch_size == 0 || oversampling == 0 ||
        (!max_rank && sketch_size <= oversampling))
        throw std::invalid_argument("invalid randomized ID grid, tolerance, batch, sketch or rank/oversampling");
    const auto spec = selector.axis_spec(axes, mo_coeff, nocc);
    if (max_rank > std::numeric_limits<std::size_t>::max()-oversampling)
        throw std::overflow_error("randomized ID rank + oversampling overflows size_t");
    // For an explicit rank the requested sketch has exactly rank + oversampling
    // rows. sketch_size remains the fallback for tolerance-only selection.
    const std::size_t requested_sketch_size = max_rank ? max_rank+oversampling : sketch_size;
    if (spec.width > std::numeric_limits<std::size_t>::max()/spec.width ||
        spec.width+1 > std::numeric_limits<std::size_t>::max()/spec.width ||
        requested_sketch_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        points.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("randomized ID dimension exceeds supported range");
    std::vector<std::array<double, 3>> kept_points;
    std::vector<double> kept_weights, kept_importance;
    std::vector<std::size_t> original;
    kept_points.reserve(points.size());
    kept_weights.reserve(points.size());
    original.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!std::isfinite(weights[i])) throw std::invalid_argument("grid weights must be finite");
        if (weights[i] == 0 || (density && point_importance[i] == 0)) continue;
        for (double x : points[i])
            if (!std::isfinite(x)) throw std::invalid_argument("grid coordinates must be finite");
        kept_points.push_back(points[i]);
        kept_weights.push_back(weights[i]);
        if (density) kept_importance.push_back(point_importance[i]);
        original.push_back(i);
    }
    if (debug && density) {
        std::size_t zero_weight = 0, zero_importance = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (weights[i] == 0) ++zero_weight;
            else if (point_importance[i] == 0) ++zero_importance;
        }
        std::cerr << "[density sketch importance] grid_points=" << points.size()
                  << " zero_weight=" << zero_weight << " zero_importance=" << zero_importance << '\n';
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
    if (n == 0) {
        if (density && max_rank) throw std::invalid_argument("requested anchor_rank exceeds eligible candidates");
        return {};
    }
    // The pair matrix has only d(d+1)/2 independent rows: drawing more
    // sketch rows cannot increase its rank. Bound dimensions and the total
    // addressable matrix size to the ILP64 QRCP backend's signed index range.
    const std::size_t physical_rank = d*(d+1)/2;
    const std::size_t physical_limit = physical_rank >
        std::numeric_limits<std::size_t>::max()-oversampling
        ? std::numeric_limits<std::size_t>::max() : physical_rank+oversampling;
    const std::size_t backend_limit =
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())/n;
    const std::size_t effective_sketch_size = requested_sketch_size <= backend_limit
        ? requested_sketch_size : std::min({requested_sketch_size, physical_limit, backend_limit});
    if (effective_sketch_size <= oversampling ||
        (max_rank && std::min(max_rank, physical_rank) > effective_sketch_size-oversampling)) {
        throw std::invalid_argument(
            "randomized ID sketch/rank exceeds the ILP64 QRCP backend limit for this grid; "
            "reduce the sketch/rank or use a tiled QRCP backend");
    }
    if (effective_sketch_size != requested_sketch_size) {
        std::cerr << "[randomized id] sketch_size reduced from " << requested_sketch_size
                  << " to " << effective_sketch_size
                  << " (ILP64 addressability/physical pair dimension; using physical pair rank + oversampling when possible)\n";
    }
    if (space == EvaluationSpace::Host)
        return select_in_space<Kokkos::DefaultHostExecutionSpace>(selector, kept_points,
            kept_weights, original, axes, tolerance, max_rank, mo_coeff, nocc,
            batch_size, debug, effective_sketch_size, oversampling, seed, audit,
            policy, axis_scales, kept_importance, ao_screening_threshold);
#ifdef ISDF_ENABLE_CUDA
    return select_in_space<Kokkos::Cuda>(selector, kept_points, kept_weights,
        original, axes, tolerance, max_rank, mo_coeff, nocc,
        batch_size, debug, effective_sketch_size, oversampling, seed, audit,
        policy, axis_scales, kept_importance, ao_screening_threshold);
#else
    return select_in_space<Kokkos::DefaultHostExecutionSpace>(selector, kept_points,
        kept_weights, original, axes, tolerance, max_rank, mo_coeff, nocc,
        batch_size, debug, effective_sketch_size, oversampling, seed, audit,
        policy, axis_scales, kept_importance, ao_screening_threshold);
#endif
}

template <class Space>
std::vector<std::size_t> RandomizedIDAnchorSelection::select_in_space(
    const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& input_points,
    const std::vector<double>& input_weights, const std::vector<std::size_t>& input_original,
    const std::vector<AxisType>& axes, double tolerance, std::size_t max_rank,
    const std::vector<double>& mo_coeff, std::size_t nocc,
    std::size_t batch_size, bool debug, std::size_t ell, std::size_t oversampling,
    std::uint64_t seed, bool audit, SketchPolicy policy,
    const std::vector<double>& axis_scales,
    const std::vector<double>& input_importance, double ao_screening_threshold)
{
    const auto started = std::chrono::steady_clock::now();
    const std::size_t d = selector.axis_spec(axes, mo_coeff, nocc).width;
    const bool density = policy == SketchPolicy::AODensity || policy == SketchPolicy::AODensityNormalized;
    const bool normalized = policy == SketchPolicy::AODensityNormalized;
    auto points = input_points;
    auto weights = input_weights;
    auto original = input_original;
    auto importance = input_importance;
    std::vector<double> ao_norms;
    if (density) {
        // A bounded eligibility pass avoids retaining invalid columns in QRCP.
        // Only O(Ngrid) norms/indices return to host, never the AO-pair matrix.
        points.clear(); weights.clear(); original.clear(); importance.clear();
        for (std::size_t start = 0; start < input_points.size();) {
            const auto count = std::min(batch_size, input_points.size()-start);
            std::vector<std::array<double,3>> batch(input_points.begin()+start, input_points.begin()+start+count);
            const auto axis = selector.evaluate_axis<Space>(batch, axes, mo_coeff, nocc, axis_scales);
            const auto phi = axis.values;
            Kokkos::View<double*, Space> norms("id_ao_norms", count);
            Kokkos::parallel_for("id_ao_norm", Kokkos::RangePolicy<Space>(0,count),
                KOKKOS_LAMBDA(const std::size_t g) {
                    double q = 0;
                    for (std::size_t a = 0; a < d; ++a) q += phi(g*d+a)*phi(g*d+a);
                    norms(g) = q;
                });
            const auto hn = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, norms);
            for (std::size_t g = 0; g < count; ++g) {
                const double q = hn(g);
                if (!std::isfinite(q)) throw std::overflow_error("non-finite AO squared norm");
                if (q <= ao_screening_threshold) continue;
                points.push_back(input_points[start+g]);
                weights.push_back(input_weights[start+g]);
                original.push_back(input_original[start+g]);
                importance.push_back(input_importance[start+g]);
                ao_norms.push_back(q);
            }
            start += count;
        }
    }
    const std::size_t n = original.size();
    if (max_rank > std::min(n, d*(d+1)/2))
        throw std::invalid_argument("requested anchor_rank exceeds eligible candidates or physical pair rank");
    if (debug && density)
        std::cerr << "[density sketch screening] candidates=" << input_points.size()
                  << " eligible=" << n << " screened_ao=" << input_points.size()-n
                  << " ao_norm2_threshold=" << ao_screening_threshold << '\n';
    if (n == 0) return {};
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
        auto axis = selector.evaluate_axis<Space>(batch, axes, mo_coeff, nocc, axis_scales);
        auto phi = axis.values;
        Kokkos::View<double*, Space> sw("id_weights", count), inv_norm("id_inv_norm", count);
        auto hw = Kokkos::create_mirror_view(sw);
        auto hi = Kokkos::create_mirror_view(inv_norm);
        for (std::size_t g = 0; g < count; ++g) {
            hw(g) = std::sqrt(density ? importance[start+g] : std::abs(weights[start+g]));
            // Normalize AO values before projection to avoid tiny-denominator
            // products. This is algebraically division of the pair column by q.
            hi(g) = normalized ? 1/std::sqrt(ao_norms[start+g]) : 1.0;
        }
        Kokkos::deep_copy(sw, hw);
        Kokkos::deep_copy(inv_norm, hi);
        const double scaling = 1/std::sqrt(static_cast<double>(ell));
        using SketchRange = Kokkos::RangePolicy<Space, Kokkos::IndexType<std::size_t>>;
        Kokkos::parallel_for("id_factored_sketch", SketchRange(0,count*ell),
            KOKKOS_LAMBDA(const std::size_t i) {
                const std::size_t g = i/ell, s = i%ell;
                double left = 0, right = 0;
                for (std::size_t a = 0; a < d; ++a) {
                    const double value = phi(g*d+a)*inv_norm(g);
                    left += value*u(a*ell+s);
                    right += value*v(a*ell+s);
                }
                y((start+g)*ell+s) = scaling*sw(g)*left*right;
            });
        std::size_t bad = 0;
        // Check this batch using local indices. Absolute sketch offsets can
        // exceed INT_MAX even when a bounded batch is small (e.g. 50k x 43k).
        Kokkos::parallel_reduce("id_check_sketch", SketchRange(0,count*ell),
            KOKKOS_LAMBDA(const std::size_t i, std::size_t& invalid) {
                if (!Kokkos::isfinite(y(start*ell+i))) ++invalid;
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
    if (max_rank && numerical_rank < max_rank)
        throw std::invalid_argument("requested anchor_rank exceeds the numerical rank of the randomized sketch");
    // A positive max_rank is an exact requested rank. Tolerance stopping is
    // used only in tolerance-driven mode (max_rank == 0).
    std::size_t rank = max_rank ? max_rank : cap;
    if (!max_rank) {
        for (std::size_t i = 0; i <= cap; ++i) {
            if (tails[i] <= tolerance*norm) { rank = i; break; }
        }
    }
    std::vector<std::size_t> selected;
    selected.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        if (jpvt[i] < 1 || static_cast<std::size_t>(jpvt[i]) > n)
            throw std::runtime_error("randomized ID QRCP returned invalid pivot");
        selected.push_back(original[static_cast<std::size_t>(jpvt[i]-1)]);
    }
    if (debug) {
        std::cerr << (normalized ? "[rand_sketch:density_col_norm]"
                                 : density ? "[rand_sketch:density]"
                                           : "[rand_sketch:quadrature]")
                  << " rank=" << rank << " sketch_size=" << ell
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
        audit_projection(selector, points, weights, axes, mo_coeff, nocc, local, axis_scales);
        if (density)
            audit_projection(selector, points, weights, axes, mo_coeff, nocc, local, axis_scales,
                importance, normalized ? ao_norms : std::vector<double>{},
                normalized ? "density_normalized" : "density_raw");
    }
    // Future full-ID continuation: retain y's packed QR, jpvt, and QR tau in a
    // result object, then solve R11 * X = R[:rank,:] in grid batches. No A^T A.
    return selected;
}

} // namespace isdf
