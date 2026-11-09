#include "anchor_select.hpp"
#include "anchor_select_qrcp.hpp"
#include "anchor_select_online_qr.hpp"
#include "anchor_select_randomized_id.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

namespace isdf {

AnchorSelection::AxisSpec AnchorSelection::axis_spec(
    const std::vector<AxisType>& axes,
    const std::vector<double>& mo_coeff,
    std::size_t nocc) const
{
    if (axes.empty()) throw std::invalid_argument("axis list must not be empty");
    const std::size_t nao = context_.nao();
    bool need_ao = false, need_grad = false, need_occ = false;
    std::size_t nfeatures = 0;
    for (AxisType axis : axes) {
        std::size_t width = 0;
        switch (axis) {
        case AxisType::AtomicOrbital: need_ao = true; width = nao; break;
        case AxisType::AtomicOrbitalGradX:
        case AxisType::AtomicOrbitalGradY:
        case AxisType::AtomicOrbitalGradZ: need_grad = true; width = nao; break;
        case AxisType::OccupiedOrbital: need_occ = true; width = nocc; break;
        default: throw std::invalid_argument("unknown AxisType");
        }
        if (width > std::numeric_limits<std::size_t>::max() - nfeatures) {
            throw std::overflow_error("axis feature count is too large");
        }
        nfeatures += width;
    }
    if (need_occ) {
        if (nao == 0 || nocc == 0 || nocc > nao ||
            nao > std::numeric_limits<std::size_t>::max() / nao ||
            mo_coeff.size() != nao * nao) {
            throw std::invalid_argument("OccupiedOrbital requires row-major (nao,nao) RHF mo_coeff and 1 <= nocc <= nao");
        }
        need_ao = true;
    }
    if (nfeatures == 0) throw std::invalid_argument("no orbital features available");
    return {nfeatures, need_ao, need_grad};
}

AnchorSelection::RuntimeAxisEvaluation AnchorSelection::evaluate_axis(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<AxisType>& axes,
    const std::vector<double>& mo_coeff, std::size_t nocc,
    EvaluationSpace space) const
{
    switch (space) {
    case EvaluationSpace::Host:
        return RuntimeAxisEvaluation(std::in_place_index<0>,
            evaluate_axis<Kokkos::DefaultHostExecutionSpace>(points, axes, mo_coeff, nocc));
    case EvaluationSpace::Auto:
#ifdef KOKKOS_ENABLE_CUDA
    case EvaluationSpace::Device:
        return RuntimeAxisEvaluation(std::in_place_index<1>,
            evaluate_axis<AnchorDeviceSpace>(points, axes, mo_coeff, nocc));
#else
        return RuntimeAxisEvaluation(std::in_place_index<0>,
            evaluate_axis<Kokkos::DefaultHostExecutionSpace>(points, axes, mo_coeff, nocc));
    case EvaluationSpace::Device:
        throw std::invalid_argument("device evaluation requires a CUDA build");
#endif
    }
    throw std::invalid_argument("unknown evaluation space");
}

std::vector<std::size_t> AnchorSelection::select(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights,
    const std::vector<AxisType>& axes,
    double tolerance,
    std::size_t max_rank,
    LocalAnchorSelectionAlgorithm algorithm,
    const std::vector<double>& mo_coeff,
    std::size_t nocc, EvaluationSpace space, std::size_t batch_size, bool debug,
    std::size_t sketch_size, std::size_t oversampling, std::uint64_t seed, bool audit) const
{
    switch (algorithm) {
    case LocalAnchorSelectionAlgorithm::QRColumnPivoting:
        return QRCPAnchorSelection::select(*this, points, weights, axes, tolerance,
                                           max_rank, mo_coeff, nocc, space);
    case LocalAnchorSelectionAlgorithm::OnlineQRColumnPivoting:
        return OnlineQRAnchorSelection::select(*this, points, weights, axes, tolerance,
                                               max_rank, mo_coeff, nocc, space, batch_size, debug);
    case LocalAnchorSelectionAlgorithm::RandomizedQRColumnPivoting:
        return RandomizedIDAnchorSelection::select(*this, points, weights, axes, tolerance,
            max_rank, mo_coeff, nocc, space, batch_size, debug,
            sketch_size, oversampling, seed, audit);
    default:
        throw std::logic_error("randomized QRCP is not implemented");
    }
}

std::vector<std::size_t> DistributedAnchorSelection::select(
    const std::vector<std::array<double, 3>>&,
    const std::vector<double>&,
    const std::vector<AxisType>&,
    DistributedAnchorSelectionAlgorithm) const
{
    throw std::logic_error("distributed SLATE SVD anchor selection is not implemented");
}

} // namespace isdf
