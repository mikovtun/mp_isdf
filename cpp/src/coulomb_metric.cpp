#include "coulomb_metric.hpp"
#include "cholesky_update.hpp"
#include "parallel_shell_tasks.hpp"

#include <blas.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
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

ERICholesky cholesky_eri(const LibcintContext& ctx, double tolerance,
                         std::size_t max_rank, bool debug)
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
    std::vector<std::pair<std::size_t, std::size_t>> shell_pairs;
    for (std::size_t s = 0; s < shells.size(); ++s)
        for (std::size_t t = 0; t <= s; ++t)
            shell_pairs.emplace_back(s, t);
    const auto& tables = ctx.tables();
    ERICholesky result;
    result.npair = npair;
    result.residual.assign(npair, 0.0);
    std::vector<std::size_t> ao_shell(nao);
    for (std::size_t s = 0; s < shells.size(); ++s)
        for (std::size_t a = shells[s].offset; a < shells[s].offset + shells[s].size; ++a)
            ao_shell[a] = s;

    // Only (mu nu | mu nu) is retained from each diagonal shell quartet.
    const auto diagonal_started = std::chrono::steady_clock::now();
    parallel_shell_tasks("isdf_eri_diagonal", shell_pairs.size(), [&](std::size_t task) {
        const auto [s, t] = shell_pairs[task];
        const auto& a = shells[s];
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
    });
    const double initial = *std::max_element(result.residual.begin(), result.residual.end());
    const double cutoff = tolerance * initial;
    const std::size_t cap = max_rank ? max_rank : npair;
    if (debug) {
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - diagonal_started).count();
        std::cerr << "[cholesky ao] shells=" << shells.size() << " pairs=" << npair
                  << " diagonal_quartets=" << shell_pairs.size()
                  << " initial_max_residual=" << initial << " cutoff=" << cutoff
                  << " diagonal_seconds=" << seconds << std::endl;
    }
    const auto first = std::max_element(result.residual.begin(), result.residual.end());
    CholeskyCandidate next{*first,
                           static_cast<std::size_t>(first-result.residual.begin()), false};
    while (result.pivots.size() < cap) {
        const double maximum = next.residual;
        if (maximum <= cutoff || maximum <= 0.0) break;
        const std::size_t pivot = next.index;
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
        const auto column_started = std::chrono::steady_clock::now();
        parallel_shell_tasks("isdf_eri_pivot_column", shell_pairs.size(),
                             [&](std::size_t task) {
            const auto [s, t] = shell_pairs[task];
            const auto& c = shells[s];
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
        });
        const auto column_finished = std::chrono::steady_clock::now();
        next = correct_cholesky_column(column, result.factors, result.residual,
                                       pivot, maximum,
                                       "Cholesky factor dimensions overflow size_t",
                                       "non-finite Cholesky vector");
        result.pivots.push_back(pivot);
        if (debug) {
            const auto integral_seconds = std::chrono::duration<double>(
                column_finished - column_started).count();
            const auto update_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - column_finished).count();
            std::cerr << "[cholesky ao] rank=" << result.pivots.size()
                      << " pivot=(" << mu << ',' << nu << ") shell_pair=("
                      << a.id << ',' << b.id << ") pivot_residual=" << maximum
                      << " max_remaining=" << next.residual
                      << " column_quartets=" << shell_pairs.size()
                      << " integral_seconds=" << integral_seconds
                      << " update_seconds=" << update_seconds << std::endl;
        }
    }
    if (debug) {
        const auto remaining = next.residual;
        std::cerr << "[cholesky ao] stop rank=" << result.pivots.size()
                  << " max_remaining=" << remaining
                  << " reason=" << (remaining == 0.0 ? "zero_residual" :
                      remaining <= cutoff ? "tolerance" : "rank_limit") << std::endl;
    }
    return result;
}

std::vector<double> coulomb_metric(const LibcintContext& ctx,
                                   const std::vector<double>& transform,
                                   std::size_t nvec, double tolerance,
                                   std::size_t max_rank, std::size_t batch_size,
                                   bool debug)
{
    const auto nao = ctx.nao();
    if (!batch_size) throw std::invalid_argument("batch_size must be positive");
    if (nao && nvec > std::numeric_limits<std::size_t>::max()/nao/nao)
        throw std::overflow_error("transform dimensions overflow size_t");
    if (transform.size() != nvec*nao*nao)
        throw std::invalid_argument("expected transform[nvec,nao*nao]");
    for (double value : transform)
        if (!std::isfinite(value)) throw std::invalid_argument("transform must be finite");
    const auto decomposition = cholesky_eri(ctx, tolerance, max_rank, debug);
    const auto npair = decomposition.npair, rank = decomposition.pivots.size();
    if (nvec && nvec > std::numeric_limits<std::size_t>::max()/nvec)
        throw std::overflow_error("metric dimensions overflow size_t");
    std::vector<double> metric(nvec*nvec, 0.0);
    if (rank && nvec > std::numeric_limits<std::size_t>::max()/rank)
        throw std::overflow_error("projection dimensions overflow size_t");
    if (rank == 0 || nvec == 0) return metric;
    if (nvec > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) ||
        npair > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) ||
        rank > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("BLAS++ metric dimensions exceed int64_t");
    std::vector<double> projection(nvec*rank, 0.0);
    const auto batches = nvec/batch_size + (nvec%batch_size != 0);
    if (debug)
        std::cerr << "[cholesky ao] projection start vectors=" << nvec
                  << " pairs=" << npair << " rank=" << rank
                  << " batches=" << batches << std::endl;
    const auto projection_started = std::chrono::steady_clock::now();
    // Pack only one batch per task. The ordered input contains both (mu,nu)
    // and (nu,mu); packed factors use just mu>=nu, so off-diagonal
    // coefficients must be summed before the GEMM.
    parallel_shell_tasks("isdf_metric_projection_batches", batches, [&](std::size_t task) {
        const auto first = task*batch_size;
        const auto rows = std::min(batch_size, nvec-first);
        if (rows > std::numeric_limits<std::size_t>::max()/npair)
            throw std::overflow_error("packed projection batch dimensions overflow size_t");
        std::vector<double> packed(rows*npair);
        for (std::size_t v = 0; v < rows; ++v)
            for (std::size_t mu = 0; mu < nao; ++mu)
                for (std::size_t nu = 0; nu <= mu; ++nu)
                    packed[v*npair+pair_index(mu,nu)] =
                        transform[(first+v)*nao*nao+mu*nao+nu] +
                        (mu == nu ? 0.0 : transform[(first+v)*nao*nao+nu*nao+mu]);
        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::Trans,
                   static_cast<int64_t>(rows), static_cast<int64_t>(rank),
                   static_cast<int64_t>(npair), 1.0, packed.data(),
                   static_cast<int64_t>(npair), decomposition.factors.data(),
                   static_cast<int64_t>(npair), 0.0, projection.data()+first*rank,
                   static_cast<int64_t>(rank));
    });
    const auto projection_finished = std::chrono::steady_clock::now();
    if (debug)
        std::cerr << "[cholesky ao] projection_seconds="
                  << std::chrono::duration<double>(projection_finished-projection_started).count()
                  << " metric_gemm start" << std::endl;
    // Rows of the Gram matrix are independent. BLAS++ dispatches each
    // GEMM to the configured host BLAS; batch tasks also use host threads
    // when that BLAS is sequential (as in the default MKL configuration).
    parallel_shell_tasks("isdf_metric_gram_batches", batches, [&](std::size_t task) {
        const auto first = task*batch_size;
        const auto rows = std::min(batch_size, nvec-first);
        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::Trans,
                   static_cast<int64_t>(rows), static_cast<int64_t>(nvec),
                   static_cast<int64_t>(rank), 1.0, projection.data()+first*rank,
                   static_cast<int64_t>(rank), projection.data(),
                   static_cast<int64_t>(rank), 0.0, metric.data()+first*nvec,
                   static_cast<int64_t>(nvec));
    });
    if (debug)
        std::cerr << "[cholesky ao] metric_gemm_seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now()-projection_finished).count()
                  << std::endl;
    return metric;
}
}
