#pragma once
#include "ao_eval.hpp"
#include <cstddef>
#include <vector>

namespace isdf {

// Packed lower-triangular AO-pair order: (mu,nu), mu >= nu,
// index mu*(mu+1)/2+nu. Factors are column-major [pivot, pair].
struct ERICholesky {
    std::size_t npair = 0;
    std::vector<double> factors;
    std::vector<std::size_t> pivots;
    std::vector<double> residual;
};

// Explicit unblocked greedy integral-direct Cholesky; tolerance is relative
// to the largest initial diagonal. max_rank=0 means no rank cap.
ERICholesky cholesky_eri(const LibcintContext& context, double tolerance,
                         std::size_t max_rank);

// transform is row-major [nvec, nao*nao] in ordered AO-pair order.
// Returns row-major [nvec, nvec], without anchor quadrature weights.
std::vector<double> coulomb_metric(const LibcintContext& context,
                                   const std::vector<double>& transform,
                                   std::size_t nvec, double tolerance,
                                   std::size_t max_rank, std::size_t batch_size);
}
