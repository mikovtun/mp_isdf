#pragma once
#include "ao_eval.hpp"
#include "cholesky_algorithm.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace isdf {
// Pair labels: 0=AO, 1/2/3=Cartesian electronic-coordinate derivatives.
// Coefficients are row-major [nvec, nao*nao], retaining ordered AO pairs.
// Returns row-major [nleft,nright], without anchor quadrature weights.
std::vector<double> derivative_cross_metric(
    const LibcintContext& context,
    const std::vector<double>& left, std::size_t nleft, std::array<int,2> left_basis,
    const std::vector<double>& right, std::size_t nright, std::array<int,2> right_basis,
    double tolerance, std::size_t max_rank, bool debug = false,
    CholeskyAlgorithm algorithm = CholeskyAlgorithm::UnblockedGreedy,
    std::uint64_t seed = 0);
}
