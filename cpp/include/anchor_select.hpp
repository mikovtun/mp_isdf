#pragma once

#include "ao_eval.hpp"
#include "enums.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <variant>
#include <vector>

namespace isdf {

class QRCPAnchorSelection;
class OnlineQRAnchorSelection;

#ifdef KOKKOS_ENABLE_CUDA
using AnchorDeviceSpace = Kokkos::Cuda;
#else
using AnchorDeviceSpace = Kokkos::DefaultHostExecutionSpace;
#endif

// Local grid-point selection by column-pivoted QR of the transpose of the
// weighted pair-product matrix. Returned indices refer to the ORIGINAL grid,
// in pivot order. Zero-weight points are excluded; negative weights enter as
// sqrt(abs(weight)). An axis list concatenates its evaluated functions before
// taking the per-point outer product with itself.
class AnchorSelection {
public:
    explicit AnchorSelection(const LibcintContext& context) : context_(context) {}
    std::size_t nao() const noexcept { return context_.nao(); }

    // Evaluated axis functions, row-major (points.size(), width). No weights or
    // pair products are applied; selection algorithms can use the values to
    // construct either a product matrix or an implicit Gram matrix.
    template <class ExecutionSpace>
    struct AxisEvaluation {
        std::size_t width;
        Kokkos::View<double*, ExecutionSpace> values;
    };
    // AO kernels and axis assembly run in ExecutionSpace; the returned view
    // stays there. Supported spaces: Kokkos::DefaultHostExecutionSpace
    // and Kokkos::Cuda when enabled.
    template <class ExecutionSpace>
    AxisEvaluation<ExecutionSpace> evaluate_axis(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<AxisType>& axes,
        const std::vector<double>& mo_coeff = {},
        std::size_t nocc = 0) const;

    using RuntimeAxisEvaluation = std::variant<
        AxisEvaluation<Kokkos::DefaultHostExecutionSpace>,
        AxisEvaluation<AnchorDeviceSpace>>;
    RuntimeAxisEvaluation evaluate_axis(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<AxisType>& axes,
        const std::vector<double>& mo_coeff, std::size_t nocc,
        EvaluationSpace space) const;

    // mo_coeff is a row-major (nao, nao) RHF matrix; nocc selects its first
    // nocc columns. Supply both if OccupiedOrbital is requested. Unused otherwise.
    // tolerance is relative to abs(R[0,0]); max_rank=0 means no explicit cap.
    std::vector<std::size_t> select(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& weights,
        const std::vector<AxisType>& axes,
        double tolerance = 1e-10,
        std::size_t max_rank = 0,
        LocalAnchorSelectionAlgorithm algorithm = LocalAnchorSelectionAlgorithm::QRColumnPivoting,
        const std::vector<double>& mo_coeff = {},
        std::size_t nocc = 0,
        EvaluationSpace space = EvaluationSpace::Auto,
        std::size_t batch_size = 128, bool debug = false) const;

private:
    friend class QRCPAnchorSelection;
    friend class OnlineQRAnchorSelection;
    struct AxisSpec {
        std::size_t width;
        bool need_ao, need_grad;
    };
    AxisSpec axis_spec(const std::vector<AxisType>& axes,
                       const std::vector<double>& mo_coeff, std::size_t nocc) const;
    const LibcintContext& context_;
};

// Distributed SVD anchor selection has no pivot-selection rule yet.
class DistributedAnchorSelection {
public:
    explicit DistributedAnchorSelection(const LibcintContext& context) : context_(context) {}
    std::vector<std::size_t> select(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& weights,
        const std::vector<AxisType>& axes,
        DistributedAnchorSelectionAlgorithm algorithm = DistributedAnchorSelectionAlgorithm::SVD) const;

private:
    const LibcintContext& context_;
};

} // namespace isdf

#include "anchor_select_impl.hpp"
