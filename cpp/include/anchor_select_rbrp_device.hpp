#pragma once
#include "anchor_select.hpp"
#include "anchor_select_rbrp.hpp"

namespace isdf {
#ifdef KOKKOS_ENABLE_CUDA
// Coordinates are the nonzero-weight grid in original ordering. Evaluates AOs
// on the GPU in bounded batches, never staging the AO grid on the host.
RBRPResult rbrp_ao_device(const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights, std::size_t batch_size,
    std::size_t block_size, double tolerance, std::size_t max_rank,
    std::uint64_t seed, double tau_block, double rank_rtol,
    std::size_t refresh_every, double svd_rtol, bool compute_id);
#endif
} // namespace isdf
