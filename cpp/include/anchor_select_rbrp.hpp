#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace isdf {

// Host AO-only robust blockwise random pivoting. AO input is row-major
// [nonzero_grid, nao]. Pair rows are generated in bounded batches; the full
// grid-by-pair matrix is never allocated. Weights may have either sign.
struct RBRPResult {
    std::vector<std::size_t> indices; // indices into the supplied AO rows
    std::vector<double> interp_vecs; // row-major [rank, grid], if compute_id
    std::vector<double> transform; // row-major [rank, nao*nao], if compute_id
    std::vector<double> history; // flattened [rank, candidates, retained, residual]
    double relative_residual = 0;
    double condition = 0;
    std::size_t retained_rank = 0;
    std::string status;
};

RBRPResult rbrp_ao_cpu(const double* ao, const double* weights,
                       std::size_t n, std::size_t nao,
                       std::size_t batch_size, std::size_t block_size,
                       double tolerance, std::size_t max_rank,
                       std::uint64_t seed,
                       double tau_block = std::numeric_limits<double>::quiet_NaN(),
                       double rank_rtol = 1e-12, std::size_t refresh_every = 8,
                       double svd_rtol = 1e-12, bool compute_id = false);

} // namespace isdf
