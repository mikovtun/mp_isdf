#pragma once

#include "enums.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace isdf {

class AnchorSelection;

// QRCP-specific selection. Kept separate from AnchorSelection so future
// algorithms can form their own data structures instead of pair products.
class QRCPAnchorSelection {
public:
    static std::vector<std::size_t> select(
        const AnchorSelection& selector,
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& weights,
        const std::vector<AxisType>& axes,
        double tolerance, std::size_t max_rank,
        const std::vector<double>& mo_coeff, std::size_t nocc,
        EvaluationSpace space);

    // Public because CUDA extended lambdas cannot reside in private methods.
    template <class ExecutionSpace>
    static std::vector<std::size_t> select_in_space(
        const AnchorSelection& selector,
        const std::vector<std::array<double, 3>>& kept_points,
        const std::vector<double>& kept_weights,
        const std::vector<std::size_t>& original,
        const std::vector<AxisType>& axes,
        double tolerance, std::size_t max_rank,
        const std::vector<double>& mo_coeff, std::size_t nocc);
};

} // namespace isdf
