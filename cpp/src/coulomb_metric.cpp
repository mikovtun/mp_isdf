#include "coulomb_metric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {
struct Shell { FINT id; std::size_t offset, size; };
std::size_t pair_index(std::size_t a, std::size_t b) { return a*(a+1)/2+b; }

std::vector<Shell> shells_for(const LibcintContext& ctx)
{
    std::vector<Shell> shells;
    std::size_t offset = 0;
    for (FINT s = 0; s < ctx.tables().nbas(); ++s) {
        const auto n = static_cast<std::size_t>(CINTcgto_spheric(s, ctx.tables().bas_data()));
        shells.push_back({s, offset, n});
        offset += n;
    }
    if (offset != ctx.nao()) throw std::runtime_error("libcint AO ordering mismatch");
    return shells;
}

// libcint stores the first shell index fastest. Initialize the output because
// screened shell quartets may leave it untouched.
std::vector<double> quartet(const LibcintWrapper& tables,
                            const std::array<FINT, 4>& ids,
                            const std::array<std::size_t, 4>& sizes)
{
    std::size_t count = 1;
    for (auto n : sizes) {
        if (n && count > std::numeric_limits<std::size_t>::max()/n)
            throw std::overflow_error("ERI shell quartet is too large");
        count *= n;
    }
    std::vector<double> values(count, 0.0);
    tables.int2e(values.data(), tables.dimensions(ids), ids);
    return values;
}
}

ERICholesky cholesky_eri(const LibcintContext& ctx, double tolerance, std::size_t max_rank)
{
    if (!std::isfinite(tolerance) || tolerance < 0)
        throw std::invalid_argument("Cholesky tolerance must be finite and nonnegative");
    const auto nao = ctx.nao();
    if (nao > 0 && nao > (std::numeric_limits<std::size_t>::max()/nao - 1)/2)
        throw std::overflow_error("AO-pair dimension overflows size_t");
    const auto npair = nao*(nao+1)/2;
    if (npair == 0) throw std::invalid_argument("empty AO basis");
    if (max_rank > npair) throw std::invalid_argument("max_rank exceeds AO-pair dimension");
    const auto shells = shells_for(ctx);
    const auto& tables = ctx.tables();
    ERICholesky result;
    result.npair = npair;
    result.residual.assign(npair, 0.0);
    std::vector<std::size_t> ao_shell(nao);
    for (std::size_t s = 0; s < shells.size(); ++s)
        for (std::size_t a = shells[s].offset; a < shells[s].offset + shells[s].size; ++a)
            ao_shell[a] = s;

    // Only (mu nu | mu nu) is retained from each diagonal shell quartet.
    for (std::size_t s = 0; s < shells.size(); ++s) {
        const auto& a = shells[s];
        for (std::size_t t = 0; t <= s; ++t) {
            const auto& b = shells[t];
            const auto block = quartet(tables, {a.id,b.id,a.id,b.id},
                                       {a.size,b.size,a.size,b.size});
            for (std::size_t i = 0; i < a.size; ++i)
                for (std::size_t j = 0; j < b.size; ++j) {
                    const auto mu = a.offset+i, nu = b.offset+j;
                    if (mu < nu) continue;
                    const auto value = block[i + a.size*(j + b.size*(i + a.size*j))];
                    if (!std::isfinite(value) || value < -1e-12)
                        throw std::runtime_error("invalid ERI diagonal");
                    result.residual[pair_index(mu,nu)] = std::max(0.0, value);
                }
        }
    }
    const double initial = *std::max_element(result.residual.begin(), result.residual.end());
    const double cutoff = tolerance * initial;
    const std::size_t cap = max_rank ? max_rank : npair;
    while (result.pivots.size() < cap) {
        const auto it = std::max_element(result.residual.begin(), result.residual.end());
        const double maximum = *it;
        if (maximum <= cutoff || maximum <= 0.0) break;
        const std::size_t pivot = static_cast<std::size_t>(it - result.residual.begin());
        // Invert the packed index without floating-point rounding.
        std::size_t lo = 0, hi = nao;
        while (lo + 1 < hi) {
            const auto mid = lo + (hi-lo)/2;
            if (pair_index(mid,0) <= pivot) lo = mid; else hi = mid;
        }
        const auto mu = lo, nu = pivot - pair_index(mu,0);
        const auto& a = shells[ao_shell[mu]];
        const auto& b = shells[ao_shell[nu]];
        std::vector<double> column(npair, 0.0);
        for (std::size_t s = 0; s < shells.size(); ++s) {
            const auto& c = shells[s];
            for (std::size_t t = 0; t <= s; ++t) {
                const auto& d = shells[t];
                const auto block = quartet(tables, {c.id,d.id,a.id,b.id},
                                           {c.size,d.size,a.size,b.size});
                const auto ai = mu-a.offset, bi = nu-b.offset;
                for (std::size_t i = 0; i < c.size; ++i)
                    for (std::size_t j = 0; j < d.size; ++j) {
                        const auto x = c.offset+i, y = d.offset+j;
                        if (x < y) continue;
                        column[pair_index(x,y)] =
                            block[i + c.size*(j + d.size*(ai + a.size*bi))];
                    }
            }
        }
        const auto rank = result.pivots.size();
        const double denom = std::sqrt(maximum);
        for (std::size_t p = 0; p < npair; ++p) {
            double value = column[p];
            for (std::size_t k = 0; k < rank; ++k)
                value -= result.factors[k*npair+p] * result.factors[k*npair+pivot];
            column[p] = value / denom;
            if (!std::isfinite(column[p])) throw std::runtime_error("non-finite Cholesky vector");
        }
        if (npair > std::numeric_limits<std::size_t>::max() / (rank + 1))
            throw std::overflow_error("Cholesky factor dimensions overflow size_t");
        result.pivots.push_back(pivot);
        result.factors.insert(result.factors.end(), column.begin(), column.end());
        for (std::size_t p = 0; p < npair; ++p)
            result.residual[p] = std::max(0.0, result.residual[p] - column[p]*column[p]);
        result.residual[pivot] = 0.0;
    }
    return result;
}

std::vector<double> coulomb_metric(const LibcintContext& ctx,
                                   const std::vector<double>& transform,
                                   std::size_t nvec, double tolerance,
                                   std::size_t max_rank, std::size_t batch_size)
{
    const auto nao = ctx.nao();
    if (!batch_size) throw std::invalid_argument("batch_size must be positive");
    if (nao && nvec > std::numeric_limits<std::size_t>::max()/nao/nao)
        throw std::overflow_error("transform dimensions overflow size_t");
    if (transform.size() != nvec*nao*nao)
        throw std::invalid_argument("expected transform[nvec,nao*nao]");
    for (double value : transform)
        if (!std::isfinite(value)) throw std::invalid_argument("transform must be finite");
    const auto decomposition = cholesky_eri(ctx, tolerance, max_rank);
    const auto npair = decomposition.npair, rank = decomposition.pivots.size();
    if (nvec && nvec > std::numeric_limits<std::size_t>::max()/nvec)
        throw std::overflow_error("metric dimensions overflow size_t");
    std::vector<double> metric(nvec*nvec, 0.0);
    if (rank && nvec > std::numeric_limits<std::size_t>::max()/rank)
        throw std::overflow_error("projection dimensions overflow size_t");
    std::vector<double> projection(nvec*rank, 0.0);
    for (std::size_t first = 0; first < nvec; first += std::min(batch_size,nvec-first)) {
        const auto end = first + std::min(batch_size,nvec-first);
        for (std::size_t v = first; v < end; ++v)
            for (std::size_t mu = 0; mu < nao; ++mu)
                for (std::size_t nu = 0; nu <= mu; ++nu) {
                    const double coeff = transform[v*nao*nao+mu*nao+nu] +
                        (mu == nu ? 0.0 : transform[v*nao*nao+nu*nao+mu]);
                    const auto p = pair_index(mu,nu);
                    for (std::size_t k = 0; k < rank; ++k)
                        projection[v*rank+k] += coeff*decomposition.factors[k*npair+p];
                }
    }
    for (std::size_t v = 0; v < nvec; ++v)
        for (std::size_t w = 0; w <= v; ++w) {
            double sum = 0.0;
            for (std::size_t k = 0; k < rank; ++k)
                sum += projection[v*rank+k]*projection[w*rank+k];
            metric[v*nvec+w] = metric[w*nvec+v] = sum;
        }
    return metric;
}
}
