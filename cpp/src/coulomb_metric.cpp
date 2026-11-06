#include "coulomb_metric.hpp"
#include "parallel_shell_tasks.hpp"

#include <blas.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace isdf {
namespace {
std::size_t pair_index(std::size_t a, std::size_t b) { return a*(a+1)/2+b; }
}

std::vector<double> coulomb_metric(const LibcintContext& ctx,
                                   const std::vector<double>& transform,
                                   std::size_t nvec, double tolerance,
                                   std::size_t max_rank, std::size_t batch_size,
                                   bool debug, CholeskyAlgorithm algorithm,
                                   std::uint64_t seed)
{
    const auto nao = ctx.nao();
    if (!batch_size) throw std::invalid_argument("batch_size must be positive");
    if (nao && nvec > std::numeric_limits<std::size_t>::max()/nao/nao)
        throw std::overflow_error("transform dimensions overflow size_t");
    if (transform.size() != nvec*nao*nao)
        throw std::invalid_argument("expected transform[nvec,nao*nao]");
    for (double value : transform)
        if (!std::isfinite(value)) throw std::invalid_argument("transform must be finite");
    const auto decomposition = algorithm == CholeskyAlgorithm::UnblockedUniform ?
        cholesky_eri_uniform(ctx, tolerance, max_rank, seed, debug) :
        cholesky_eri(ctx, tolerance, max_rank, debug);
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
    // Pack only one batch per task. Ordered off-diagonal coefficients must be
    // folded before multiplying by the packed AO-pair Cholesky factors.
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
    // Independent output rows permit host parallelism even with sequential BLAS.
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
} // namespace isdf
