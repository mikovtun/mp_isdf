#include "anchor_select_rbrp.hpp"

#include <lapack.hh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {

class PairRows {
public:
    PairRows(const double* ao, const double* w, std::size_t n,
             std::size_t width, std::size_t batch)
        : ao_(ao), w_(w), n_(n), width_(width), p_(width*(width+1)/2), batch_(batch) {}
    std::size_t width() const { return width_; }
    std::size_t pairs() const { return p_; }
    std::size_t size() const { return n_; }
    double sqrtw(std::size_t g) const { return std::sqrt(std::abs(w_[g])); }
    double weight(std::size_t g) const { return w_[g]; }
    void row(std::size_t g, double* dst) const {
        std::size_t t = 0;
        const double s = sqrtw(g);
        for (std::size_t mu = 0; mu < width_; ++mu)
            for (std::size_t nu = mu; nu < width_; ++nu)
                dst[t++] = s * ao_[g*width_+mu] * ao_[g*width_+nu] *
                           (mu == nu ? 1.0 : std::sqrt(2.0));
    }
    void for_batches(const auto& fn) const {
        std::vector<double> rows(std::min(batch_, n_)*p_);
        for (std::size_t start = 0; start < n_; start += batch_) {
            const auto count = std::min(batch_, n_-start);
            for (std::size_t g = 0; g < count; ++g) row(start+g, rows.data()+g*p_);
            fn(start, count, rows.data());
        }
    }
    std::vector<double> norm2() const {
        std::vector<double> d(n_);
        for (std::size_t g = 0; g < n_; ++g) {
            double sum = 0;
            for (std::size_t j = 0; j < width_; ++j)
                sum += ao_[g*width_+j]*ao_[g*width_+j];
            d[g] = std::abs(w_[g])*sum*sum;
            if (!std::isfinite(d[g])) throw std::overflow_error("non-finite RBRP AO pair norm");
        }
        return d;
    }
private:
    const double* ao_;
    const double* w_;
    std::size_t n_, width_, p_, batch_;
};

double sum(const std::vector<double>& a) {
    return std::accumulate(a.begin(), a.end(), 0.0);
}

// Q is column-major [p,k], L row-major [n,k]. Refresh from explicit pair rows.
std::vector<double> residual(const PairRows& op, const std::vector<double>& q,
                             const std::vector<double>& l, std::size_t rank,
                             const std::vector<unsigned char>& chosen) {
    const auto n = op.size(), p = op.pairs();
    std::vector<double> d(n);
    op.for_batches([&](std::size_t start, std::size_t count, const double* rows) {
        for (std::size_t g = 0; g < count; ++g) {
            const auto id = start+g;
            if (chosen[id]) { d[id] = 0; continue; }
            double value = 0;
            for (std::size_t i = 0; i < p; ++i) {
                double e = rows[g*p+i];
                for (std::size_t t = 0; t < rank; ++t) e -= l[id*rank+t]*q[t*p+i];
                value += e*e;
            }
            d[id] = value;
        }
    });
    return d;
}

void append_block(const PairRows& op, std::vector<double>& q, std::vector<double>& l,
                  std::vector<double>& d, const std::vector<unsigned char>& previously,
                  const double* qb, std::size_t old, std::size_t keep) {
    const auto n = op.size(), p = op.pairs();
    std::vector<double> next(n*(old+keep));
    for (std::size_t g = 0; g < n; ++g)
        for (std::size_t t = 0; t < old; ++t)
            next[g*(old+keep)+t] = l[g*old+t];
    op.for_batches([&](std::size_t start, std::size_t count, const double* rows) {
        for (std::size_t g = 0; g < count; ++g) {
            const auto id = start+g;
            double decrease = 0;
            for (std::size_t t = 0; t < keep; ++t) {
                double dot = 0;
                if (!previously[id])
                    for (std::size_t i = 0; i < p; ++i) dot += rows[g*p+i]*qb[t*p+i];
                next[id*(old+keep)+old+t] = dot;
                decrease += dot*dot;
            }
            d[id] = previously[id] ? 0.0 : std::max(0.0, d[id]-decrease);
        }
    });
    l.swap(next);
    q.insert(q.end(), qb, qb+keep*p);
}

// Reorthogonalize local Q after the LAPACK QR and discard no columns: the
// rank cutoff on R has already removed numerically dependent directions.
void reorthogonalize(double* qb, const std::vector<double>& q,
                     std::size_t p, std::size_t old, std::size_t keep) {
    for (std::size_t j = 0; j < keep; ++j) {
        double* col = qb+j*p;
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t t = 0; t < old; ++t) {
                double dot = 0;
                for (std::size_t i = 0; i < p; ++i) dot += q[t*p+i]*col[i];
                for (std::size_t i = 0; i < p; ++i) col[i] -= dot*q[t*p+i];
            }
            for (std::size_t t = 0; t < j; ++t) {
                double dot = 0;
                for (std::size_t i = 0; i < p; ++i) dot += qb[t*p+i]*col[i];
                for (std::size_t i = 0; i < p; ++i) col[i] -= dot*qb[t*p+i];
            }
        }
        double norm = 0;
        for (std::size_t i = 0; i < p; ++i) norm = std::hypot(norm, col[i]);
        if (!(norm > 0) || !std::isfinite(norm))
            throw std::runtime_error("RBRP block lost orthogonality");
        for (std::size_t i = 0; i < p; ++i) col[i] /= norm;
    }
}

void make_id(const PairRows& op, const std::vector<double>& q,
             const std::vector<double>& l, RBRPResult& out, double svd_rtol) {
    const std::size_t n = op.size(), p = op.pairs(), k = out.indices.size();
    if (!k) return;
    std::vector<double> a(k*k), u(k*k), vt(k*k), s(k);
    for (std::size_t col = 0; col < k; ++col)
        for (std::size_t row = 0; row < k; ++row)
            a[row+col*k] = l[out.indices[row]*k+col];
    if (lapack::gesdd(lapack::Job::AllVec, static_cast<std::int64_t>(k),
                      static_cast<std::int64_t>(k), a.data(), k, s.data(),
                      u.data(), k, vt.data(), k) != 0)
        throw std::runtime_error("RBRP small L1 SVD failed");
    out.condition = s.back() ? s.front()/s.back() : std::numeric_limits<double>::infinity();
    for (double x : s) if (x > svd_rtol*s.front()) ++out.retained_rank;
    if (!out.retained_rank) throw std::runtime_error("RBRP anchor block has zero numerical rank");
    // inv = V diag(1/s) U^T, row-major [k,k].
    std::vector<double> inv(k*k, 0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            for (std::size_t t = 0; t < out.retained_rank; ++t)
                inv[i*k+j] += vt[t+i*k]*u[j+t*k]/s[t];
    out.interp_vecs.resize(k*n);
    for (std::size_t g = 0; g < n; ++g)
        for (std::size_t aidx = 0; aidx < k; ++aidx) {
            double h = 0;
            for (std::size_t t = 0; t < k; ++t) h += l[g*k+t]*inv[t*k+aidx];
            out.interp_vecs[aidx*n+g] = h * op.sqrtw(out.indices[aidx]) /
                (op.weight(out.indices[aidx])*op.sqrtw(g));
        }
    // Paper's cardinality condition is imposed even after SVD clipping.
    for (std::size_t aidx = 0; aidx < k; ++aidx)
        for (std::size_t b = 0; b < k; ++b)
            out.interp_vecs[aidx*n+out.indices[b]] =
                aidx == b ? 1.0/op.weight(out.indices[b]) : 0.0;
    const std::size_t width = op.width();
    out.transform.assign(k*width*width, 0);
    std::size_t packed = 0;
    for (std::size_t mu = 0; mu < width; ++mu)
        for (std::size_t nu = mu; nu < width; ++nu, ++packed)
            for (std::size_t aidx = 0; aidx < k; ++aidx) {
                double value = 0;
                for (std::size_t t = 0; t < k; ++t)
                    value += inv[t*k+aidx]*q[t*p+packed];
                value *= op.sqrtw(out.indices[aidx]) / op.weight(out.indices[aidx]);
                if (mu != nu) value /= std::sqrt(2.0);
                out.transform[aidx*width*width+mu*width+nu] = value;
                out.transform[aidx*width*width+nu*width+mu] = value;
            }
}

} // namespace

RBRPResult rbrp_ao_cpu(const double* ao, const double* weights,
                       std::size_t n, std::size_t nao, std::size_t batch_size,
                       std::size_t block_size, double tolerance,
                       std::size_t max_rank, std::uint64_t seed,
                       double tau_block, double rank_rtol,
                       std::size_t refresh_every, double svd_rtol, bool compute_id) {
    if (!n || !nao || !batch_size || !block_size ||
        nao == std::numeric_limits<std::size_t>::max() ||
        nao > std::numeric_limits<std::size_t>::max()/(nao+1) ||
        n > std::numeric_limits<std::size_t>::max()/nao ||
        !std::isfinite(tolerance) || tolerance < 0 ||
        !std::isfinite(rank_rtol) || rank_rtol <= 0 ||
        !std::isfinite(svd_rtol) || svd_rtol < 0 ||
        (std::isfinite(tau_block) && (tau_block < 0 || tau_block >= 1)) ||
        (!std::isfinite(tau_block) && !std::isnan(tau_block)))
        throw std::invalid_argument("invalid RBRP dimensions, batch size or tolerances");
    const auto p = nao*(nao+1)/2;
    if (std::min(batch_size, n) > std::numeric_limits<std::size_t>::max()/p)
        throw std::overflow_error("RBRP pair batch exceeds addressable storage");
    const auto cap = std::min({p, n, max_rank ? max_rank : std::min(n,p)});
    if (compute_id && (nao > std::numeric_limits<std::size_t>::max()/nao ||
                       cap > std::numeric_limits<std::size_t>::max()/(nao*nao)))
        throw std::overflow_error("RBRP ordered-pair transform exceeds addressable storage");
    if (p > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        block_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        cap && (n > std::numeric_limits<std::size_t>::max()/cap ||
                p > std::numeric_limits<std::size_t>::max()/cap))
        throw std::overflow_error("RBRP dimensions exceed supported range");
    if (std::isnan(tau_block)) tau_block = block_size > 1 ? 1.0/block_size : 0.0;
    for (std::size_t g = 0; g < n; ++g) {
        if (!std::isfinite(weights[g]) || weights[g] == 0)
            throw std::invalid_argument("RBRP requires finite nonzero weights");
        for (std::size_t j = 0; j < nao; ++j)
            if (!std::isfinite(ao[g*nao+j]))
                throw std::invalid_argument("RBRP requires finite real AOs");
    }
    PairRows op(ao, weights, n, nao, batch_size);
    auto d = op.norm2();
    const double total = sum(d), target = tolerance*tolerance*total;
    if (!std::isfinite(total)) throw std::overflow_error("non-finite RBRP total norm");
    std::vector<double> q, l;
    std::vector<unsigned char> chosen(n, 0);
    std::mt19937_64 rng(seed);
    RBRPResult out;
    out.status = "rank_cap";
    std::size_t iteration = 0;
    while (true) {
        const auto old = out.indices.size();
        if (old && (sum(d) <= target || (refresh_every && iteration % refresh_every == 0)))
            d = residual(op, q, l, old, chosen);
        if (sum(d) <= target) { out.status = "tolerance"; break; }
        if (old >= cap) break;
        std::vector<std::size_t> active;
        for (std::size_t i = 0; i < n; ++i) if (d[i] > 0 && !chosen[i]) active.push_back(i);
        if (active.empty()) { out.status = "no_candidates"; break; }
        const auto b = std::min({block_size, cap-old, active.size()});
        std::vector<std::size_t> sample;
        sample.reserve(b);
        for (std::size_t j = 0; j < b; ++j) {
            std::vector<double> weights_left;
            weights_left.reserve(active.size());
            for (auto id : active) weights_left.push_back(d[id]);
            std::discrete_distribution<std::size_t> dist(weights_left.begin(), weights_left.end());
            const auto pos = dist(rng);
            sample.push_back(active[pos]);
            active.erase(active.begin()+static_cast<std::ptrdiff_t>(pos));
        }
        std::vector<double> v(p*b), raw(p);
        double original2 = 0;
        for (std::size_t j = 0; j < b; ++j) {
            op.row(sample[j], raw.data());
            for (std::size_t i = 0; i < p; ++i) {
                v[j*p+i] = raw[i];
                original2 += raw[i]*raw[i];
            }
        }
        // Two projection passes limit loss of orthogonality near rank deficiency.
        for (int pass = 0; pass < 2; ++pass)
            for (std::size_t j = 0; j < b; ++j)
                for (std::size_t t = 0; t < old; ++t) {
                    double dot = 0;
                    for (std::size_t i = 0; i < p; ++i) dot += v[j*p+i]*q[t*p+i];
                    for (std::size_t i = 0; i < p; ++i) v[j*p+i] -= dot*q[t*p+i];
                }
        std::vector<std::int64_t> piv(b, 0);
        std::vector<double> tau(b);
        if (lapack::geqp3(static_cast<std::int64_t>(p), static_cast<std::int64_t>(b),
                          v.data(), static_cast<std::int64_t>(p), piv.data(), tau.data()) != 0)
            throw std::runtime_error("RBRP block geqp3 failed");
        std::vector<double> tails(b, 0);
        for (std::size_t row = b; row-- > 0;) {
            tails[row] = row+1 < b ? tails[row+1] : 0;
            for (std::size_t col = row; col < b; ++col)
                tails[row] += v[col*p+row]*v[col*p+row];
        }
        if (!(tails[0] > 0) || !std::isfinite(tails[0])) {
            out.status = "numerical_rank"; break;
        }
        std::size_t keep = 0;
        for (std::size_t i = 0; i < b; ++i)
            if (tails[i] >= tau_block*tails[0]) keep = i+1;
        const double cutoff = rank_rtol*std::sqrt(original2);
        for (std::size_t i = 0; i < keep; ++i)
            if (std::abs(v[i*p+i]) <= cutoff) { keep = i; break; }
        if (!keep) { out.status = "numerical_rank"; break; }
        if (lapack::orgqr(static_cast<std::int64_t>(p), static_cast<std::int64_t>(keep),
                          static_cast<std::int64_t>(keep), v.data(),
                          static_cast<std::int64_t>(p), tau.data()) != 0)
            throw std::runtime_error("RBRP block orgqr failed");
        reorthogonalize(v.data(), q, p, old, keep);
        append_block(op, q, l, d, chosen, v.data(), old, keep);
        for (std::size_t j = 0; j < keep; ++j) {
            if (piv[j] < 1 || static_cast<std::size_t>(piv[j]) > b)
                throw std::runtime_error("RBRP block returned invalid pivot");
            const auto id = sample[static_cast<std::size_t>(piv[j]-1)];
            out.indices.push_back(id);
            chosen[id] = 1;
            d[id] = 0;
        }
        out.history.insert(out.history.end(), {
            static_cast<double>(out.indices.size()), static_cast<double>(b),
            static_cast<double>(keep), total ? std::sqrt(sum(d)/total) : 0.0});
        ++iteration;
    }
    d = residual(op, q, l, out.indices.size(), chosen);
    out.relative_residual = total ? std::sqrt(sum(d)/total) : 0;
    if (out.relative_residual <= tolerance) out.status = "tolerance";
    else if (out.status == "tolerance") out.status = "roundoff_limited";
    if (compute_id) make_id(op, q, l, out, svd_rtol);
    return out;
}

} // namespace isdf
