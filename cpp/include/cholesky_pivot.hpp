#pragma once

#include "cholesky_algorithm.hpp"
#include "cholesky_update.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace isdf {

// The maximum residual governs convergence for BOTH algorithms. Uniform
// sampling chooses among all strictly eligible, positive residuals instead of
// weighting by their magnitude. The selected pivot may not be the maximum.
inline CholeskyCandidate choose_cholesky_pivot(
    const std::vector<double>& residual, CholeskyCandidate maximum,
    double cutoff, CholeskyAlgorithm algorithm, std::mt19937_64& generator)
{
    if (maximum.residual <= cutoff || maximum.residual <= 0.0 ||
        algorithm == CholeskyAlgorithm::UnblockedGreedy)
        return maximum;
    if (algorithm != CholeskyAlgorithm::UnblockedUniform)
        throw std::invalid_argument("unknown Cholesky algorithm");
    std::size_t eligible = 0;
    for (double value : residual)
        if (value > cutoff && value > 0.0) ++eligible;
    if (!eligible) throw std::logic_error("positive Cholesky residual has no eligible pivot");
    std::uniform_int_distribution<std::size_t> pick(0, eligible - 1);
    auto ordinal = pick(generator);
    for (std::size_t p = 0; p < residual.size(); ++p)
        if (residual[p] > cutoff && residual[p] > 0.0 && ordinal-- == 0)
            return {residual[p], p, false};
    throw std::logic_error("uniform Cholesky pivot selection failed");
}

} // namespace isdf
