#pragma once

#include "coulomb_metric.hpp"
#include "cholesky_pivot.hpp"

#include <cstdint>

namespace isdf {
// Shared integral-direct implementation, selected by the two public wrappers.
ERICholesky cholesky_eri_impl(const LibcintContext& context, double tolerance,
                              std::size_t max_rank, bool debug,
                              CholeskyAlgorithm algorithm, std::uint64_t seed);
}
