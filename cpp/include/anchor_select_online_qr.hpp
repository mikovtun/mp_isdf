#pragma once

#include "anchor_select.hpp"

namespace isdf {

// Multipass, bounded-batch Householder QRCP of weighted AO-pair columns.
class OnlineQRAnchorSelection {
public:
    static std::vector<std::size_t> select(
        const AnchorSelection& selector,
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& weights,
        const std::vector<AxisType>& axes,
        double tolerance, std::size_t max_rank,
        const std::vector<double>& mo_coeff, std::size_t nocc,
        EvaluationSpace space, std::size_t batch_size, bool debug);

    // CUDA extended lambdas require a public enclosing function.
    template <class ExecutionSpace>
    static std::vector<std::size_t> select_in_space(
        const AnchorSelection& selector,
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& weights,
        const std::vector<std::size_t>& original,
        const std::vector<AxisType>& axes,
        double tolerance, std::size_t max_rank,
        const std::vector<double>& mo_coeff, std::size_t nocc,
        std::size_t batch_size, bool debug);
};

} // namespace isdf
