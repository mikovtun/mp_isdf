#include "anchor_select_online_qr.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {

template <class Space>
class ColumnStream {
public:
    ColumnStream(const AnchorSelection& selector,
                 const std::vector<std::array<double, 3>>& points,
                 const std::vector<double>& weights,
                 const std::vector<AxisType>& axes,
                 const std::vector<double>& coeff, std::size_t nocc,
                 std::size_t width)
        : selector_(selector), points_(points), weights_(weights), axes_(axes),
          coeff_(coeff), nocc_(nocc), width_(width), rows_(width*width) {}

    // Each call evaluates only the requested grid batch, on the selected space.
    Kokkos::View<double*, Space> columns(const std::vector<std::size_t>& ids) const
    {
        std::vector<std::array<double, 3>> batch;
        batch.reserve(ids.size());
        Kokkos::View<double*, Space> w("online_weights", ids.size());
        auto hw = Kokkos::create_mirror_view(w);
        for (std::size_t j = 0; j < ids.size(); ++j) {
            batch.push_back(points_[ids[j]]);
            hw(j) = std::sqrt(std::abs(weights_[ids[j]]));
        }
        Kokkos::deep_copy(w, hw);
        auto axis = selector_.evaluate_axis<Space>(batch, axes_, coeff_, nocc_);
        auto features = axis.values;
        Kokkos::View<double*, Space> a("online_pair_batch", ids.size()*rows_);
        const std::size_t m = rows_, width = width_;
        Kokkos::parallel_for("online_pair_columns", Kokkos::RangePolicy<Space>(0, ids.size()*m),
            KOKKOS_LAMBDA(const std::size_t i) {
                const std::size_t g = i/m, pair = i%m;
                a(i) = w(g)*features(g*width + pair/width)*features(g*width + pair%width);
            });
        return a;
    }

    // Apply existing reflectors in their construction order: H[k-1]...H[0] A.
    void transform(Kokkos::View<double*, Space> a, std::size_t count,
                   const std::vector<Kokkos::View<double*, Space>>& v,
                   const std::vector<double>& tau) const
    {
        const std::size_t m = rows_;
        for (std::size_t j = 0; j < v.size(); ++j) {
            auto reflector = v[j];
            const double t = tau[j];
            Kokkos::parallel_for("online_apply_reflector", Kokkos::RangePolicy<Space>(0, count),
                KOKKOS_LAMBDA(const std::size_t g) {
                    double dot = 0;
                    for (std::size_t i = j; i < m; ++i) dot += reflector(i)*a(g*m+i);
                    dot *= t;
                    for (std::size_t i = j; i < m; ++i) a(g*m+i) -= reflector(i)*dot;
                });
        }
    }

    std::vector<double> norms(Kokkos::View<double*, Space> a, std::size_t count,
                              std::size_t first) const
    {
        const std::size_t m = rows_;
        Kokkos::View<double*, Space> d_norm("online_batch_norms", count);
        Kokkos::parallel_for("online_tail_norms", Kokkos::RangePolicy<Space>(0, count),
            KOKKOS_LAMBDA(const std::size_t g) {
                double scale = 0, ssq = 1;
                for (std::size_t i = first; i < m; ++i) {
                    const double x = Kokkos::abs(a(g*m+i));
                    if (x != 0) {
                        if (scale < x) { const double r = scale/x; ssq = 1+ssq*r*r; scale = x; }
                        else { const double r = x/scale; ssq += r*r; }
                    }
                }
                d_norm(g) = scale*Kokkos::sqrt(ssq);
            });
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_norm);
        std::vector<double> result(count);
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(h(i))) throw std::overflow_error("non-finite online AO pair column");
            result[i] = h(i);
        }
        return result;
    }

    std::size_t rows() const { return rows_; }

private:
    const AnchorSelection& selector_;
    const std::vector<std::array<double, 3>>& points_;
    const std::vector<double>& weights_;
    const std::vector<AxisType>& axes_;
    const std::vector<double>& coeff_;
    std::size_t nocc_, width_, rows_;
};

// Stable Householder representation with v[j]=1 and Hx=[beta,0,...].
void make_reflector(std::vector<double>& column, std::size_t j, double& tau)
{
    double tail = 0;
    for (std::size_t i = j+1; i < column.size(); ++i)
        tail = std::hypot(tail, column[i]);
    const double alpha = column[j];
    if (tail == 0) { tau = 0; column[j] = 1; return; }
    const double beta = -std::copysign(std::hypot(alpha, tail), alpha);
    tau = (beta-alpha)/beta;
    const double divisor = alpha-beta;
    column[j] = 1;
    for (std::size_t i = j+1; i < column.size(); ++i) column[i] /= divisor;
}

} // namespace

std::vector<std::size_t> OnlineQRAnchorSelection::select(
    const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights, const std::vector<AxisType>& axes,
    double tolerance, std::size_t max_rank,
    const std::vector<double>& mo_coeff, std::size_t nocc,
    EvaluationSpace space, std::size_t batch_size, bool debug)
{
    if (points.size() != weights.size() || !std::isfinite(tolerance) || tolerance < 0 ||
        batch_size == 0) throw std::invalid_argument("invalid online QR weights, tolerance or batch size");
    // Validate the axis specification even if the grid is empty.
    const auto spec = selector.axis_spec(axes, mo_coeff, nocc);
    if (spec.width > std::numeric_limits<std::size_t>::max()/spec.width ||
        spec.width*spec.width > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("AO pair dimension is too large");
    std::vector<std::size_t> original;
    original.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!std::isfinite(weights[i])) throw std::invalid_argument("grid weights must be finite");
        if (weights[i] == 0) continue;
        for (double x : points[i])
            if (!std::isfinite(x)) throw std::invalid_argument("grid coordinates must be finite");
        original.push_back(i);
    }
    if (space == EvaluationSpace::Device) {
#ifndef ISDF_ENABLE_CUDA
        throw std::invalid_argument("device evaluation requires a CUDA build");
#endif
    } else if (space != EvaluationSpace::Auto && space != EvaluationSpace::Host) {
        throw std::invalid_argument("unknown evaluation space");
    }
    if (original.empty()) return {};
    if (std::min(batch_size, original.size()) >
        std::numeric_limits<std::size_t>::max()/(spec.width*spec.width))
        throw std::overflow_error("online QR batch is too large");
    if (space == EvaluationSpace::Host)
        return select_in_space<Kokkos::DefaultHostExecutionSpace>(
            selector, points, weights, original, axes, tolerance, max_rank, mo_coeff, nocc, batch_size, debug);
#ifdef ISDF_ENABLE_CUDA
    return select_in_space<Kokkos::Cuda>(
        selector, points, weights, original, axes, tolerance, max_rank, mo_coeff, nocc, batch_size, debug);
#else
    return select_in_space<Kokkos::DefaultHostExecutionSpace>(
        selector, points, weights, original, axes, tolerance, max_rank, mo_coeff, nocc, batch_size, debug);
#endif
}

template <class Space>
std::vector<std::size_t> OnlineQRAnchorSelection::select_in_space(
    const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights, const std::vector<std::size_t>& original,
    const std::vector<AxisType>& axes, double tolerance, std::size_t max_rank,
    const std::vector<double>& mo_coeff, std::size_t nocc, std::size_t batch_size, bool debug)
{
    const auto started = std::chrono::steady_clock::now();
    const std::size_t width = selector.axis_spec(axes, mo_coeff, nocc).width;
    ColumnStream<Space> stream(selector, points, weights, axes, mo_coeff, nocc, width);
    const std::size_t m = stream.rows(), n = original.size();
    const std::size_t cap = std::min({m, n, max_rank ? max_rank : std::min(m,n)});
    std::vector<double> estimated(n), reference(n), initial(n);
    std::vector<unsigned char> selected(n, 0);
    std::vector<Kokkos::View<double*, Space>> reflectors;
    std::vector<double> taus;
    std::vector<std::size_t> result;
    reflectors.reserve(cap);
    taus.reserve(cap);
    result.reserve(cap);
    std::size_t full_scans = 0, candidate_checks = 0, reselections = 0;
    std::size_t norm_updates = 0, refresh_columns = 0;
    auto scan = [&](bool direct) {
        if (direct) ++full_scans;
        for (std::size_t start = 0; start < n; start += std::min(batch_size,n-start)) {
            const std::size_t count = std::min(batch_size,n-start);
            std::vector<std::size_t> ids(original.begin()+start, original.begin()+start+count);
            auto a = stream.columns(ids);
            if (direct) stream.transform(a, count, reflectors, taus);
            auto norms = stream.norms(a, count, reflectors.size());
            for (std::size_t g = 0; g < count; ++g) {
                estimated[start+g] = selected[start+g] ? 0 : norms[g];
                reference[start+g] = estimated[start+g];
            }
        }
    };
    scan(false);
    initial = estimated;
    const double first = *std::max_element(initial.begin(), initial.end());
    double frob = 0;
    for (double x : initial) frob = std::hypot(frob, x);
    const double floor = 32*std::numeric_limits<double>::epsilon()*frob;
    const double refresh_threshold = std::sqrt(std::numeric_limits<double>::epsilon());
    const double gamma = (m*std::numeric_limits<double>::epsilon() < 0.5)
        ? m*std::numeric_limits<double>::epsilon()/(1-m*std::numeric_limits<double>::epsilon()) : 1.0;

    auto winner = [&]() {
        std::size_t p = 0;
        for (std::size_t g = 1; g < n; ++g)
            if (estimated[g] > estimated[p]) p = g;
        return p;
    };
    while (true) {
        std::size_t p = winner();
        if (estimated[p] <= tolerance*first || result.size() == cap || estimated[p] <= floor) {
            scan(true); // No optimistic estimate may terminate the algorithm.
            p = winner();
            if (estimated[p] <= tolerance*first || result.size() == cap || estimated[p] <= floor)
                break;
        }
        // Directly verify the leading candidate; if stale, reselect.
        std::vector<double> column;
        for (;;) {
            p = winner();
            auto a = stream.columns({original[p]});
            stream.transform(a, 1, reflectors, taus);
            const double norm = stream.norms(a, 1, reflectors.size())[0];
            ++candidate_checks;
            estimated[p] = reference[p] = norm;
            if (p == winner()) {
                auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
                column.resize(m);
                for (std::size_t i = 0; i < m; ++i) column[i] = h(i);
                break;
            }
            ++reselections;
        }
        if (estimated[p] <= floor || estimated[p] <= tolerance*first) {
            scan(true);
            if (estimated[winner()] <= floor || estimated[winner()] <= tolerance*first) break;
            continue;
        }
        const std::size_t k = reflectors.size();
        double tau = 0;
        make_reflector(column, k, tau);
        for (std::size_t i = 0; i < k; ++i) column[i] = 0;
        Kokkos::View<double*, Space> v("online_reflector", m);
        auto h_v = Kokkos::create_mirror_view(v);
        for (std::size_t i = 0; i < m; ++i) h_v(i) = column[i];
        Kokkos::deep_copy(v, h_v);
        reflectors.push_back(v);
        taus.push_back(tau);
        result.push_back(original[p]);
        selected[p] = 1;
        estimated[p] = reference[p] = 0;

        // q_k = H[0] ... H[k] e_k. Apply reflectors in reverse order.
        Kokkos::View<double*, Space> q("online_q", m);
        Kokkos::deep_copy(q, 0.0);
        Kokkos::parallel_for("online_q_seed", Kokkos::RangePolicy<Space>(k, k+1),
            KOKKOS_LAMBDA(const std::size_t i) { q(i) = 1; });
        for (std::size_t rev = k+1; rev > 0; --rev) {
            const std::size_t j = rev-1;
            auto r = reflectors[j];
            double dot = 0;
            Kokkos::parallel_reduce("online_q_dot", Kokkos::RangePolicy<Space>(j,m),
                KOKKOS_LAMBDA(const std::size_t i, double& sum) { sum += r(i)*q(i); }, dot);
            const double factor = taus[j]*dot;
            Kokkos::parallel_for("online_q_update", Kokkos::RangePolicy<Space>(j,m),
                KOKKOS_LAMBDA(const std::size_t i) { q(i) -= factor*r(i); });
        }
        for (std::size_t start = 0; start < n; start += std::min(batch_size,n-start)) {
            const std::size_t count = std::min(batch_size,n-start);
            std::vector<std::size_t> ids(original.begin()+start, original.begin()+start+count);
            auto a = stream.columns(ids);
            Kokkos::View<double*, Space> d_row("online_r_row", count);
            Kokkos::View<double*, Space> d_abs("online_r_dot_abs", count);
            Kokkos::parallel_for("online_row", Kokkos::RangePolicy<Space>(0,count),
                KOKKOS_LAMBDA(const std::size_t g) {
                    double dot = 0, absdot = 0;
                    for (std::size_t i = 0; i < m; ++i) {
                        const double term = q(i)*a(g*m+i);
                        dot += term;
                        absdot += Kokkos::abs(term);
                    }
                    d_row(g) = dot;
                    d_abs(g) = absdot;
                });
            auto row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_row);
            auto absrow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_abs);
            std::vector<std::size_t> refresh;
            for (std::size_t g = 0; g < count; ++g) {
                const std::size_t id = start+g;
                if (selected[id]) continue;
                ++norm_updates;
                const double old = estimated[id], r = std::abs(row(g));
                if (!std::isfinite(r) || !std::isfinite(absrow(g)))
                    throw std::overflow_error("non-finite online QR coefficient");
                const double ratio = old > 0 ? r/old : 0;
                const double temp = std::max(0.0, (1+ratio)*(1-ratio));
                const double next = old*std::sqrt(temp);
                const double ref_ratio = reference[id] > 0 ? old/reference[id] : 0;
                const double err = gamma*absrow(g);
                estimated[id] = next;
                if (old == 0 || ratio > 1 || temp*ref_ratio*ref_ratio <= refresh_threshold ||
                    (2*r*err+err*err) > refresh_threshold*next*next)
                    refresh.push_back(original[id]);
            }
            if (!refresh.empty()) {
                refresh_columns += refresh.size();
                auto b = stream.columns(refresh);
                stream.transform(b, refresh.size(), reflectors, taus);
                auto norms = stream.norms(b, refresh.size(), k+1);
                for (std::size_t j = 0; j < refresh.size(); ++j) {
                    // original is ordered and each batch is a subset of [start,start+count).
                    const auto id = static_cast<std::size_t>(
                        std::lower_bound(original.begin()+start, original.begin()+start+count, refresh[j])
                        - original.begin());
                    estimated[id] = reference[id] = norms[j];
                }
            }
        }
    }
    if (debug) {
        const double residual = *std::max_element(estimated.begin(), estimated.end());
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now()-started).count();
        std::cerr << "[online qrcp] rank=" << result.size() << " rows=" << m
                  << " columns=" << n << " batch_size=" << batch_size
                  << " max_initial_norm=" << first << " max_residual=" << residual
                  << " relative_residual=" << (first > 0 ? residual/first : 0)
                  << " full_scans=" << full_scans
                  << " candidate_checks=" << candidate_checks
                  << " candidate_reselections=" << reselections
                  << " norm_updates=" << norm_updates
                  << " refreshed_columns=" << refresh_columns
                  << " elapsed=" << elapsed << " s\n";
    }
    return result;
}

} // namespace isdf
