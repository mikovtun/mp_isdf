#include "anchor_select.hpp"
#include "anchor_select_qrcp.hpp"
#include "anchor_select_online_qr.hpp"
#include "anchor_select_randomized_id.hpp"
#include "anchor_select_cvt.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
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
    std::size_t sketch_size, std::size_t oversampling, std::uint64_t seed, bool audit,
    const std::vector<double>& cvt_weights, std::size_t cvt_max_iter, double cvt_tol,
    const std::vector<double>& axis_weights,
    const std::vector<double>& point_importance, double ao_screening_threshold) const
{
    const bool density_sketch =
        algorithm == LocalAnchorSelectionAlgorithm::RandomizedAODensityKhatriRao ||
        algorithm == LocalAnchorSelectionAlgorithm::RandomizedAODensityNormalizedKhatriRao;
    if (density_sketch && (axes.size() != 1 || axes[0] != AxisType::AtomicOrbital))
        throw std::invalid_argument("density Khatri-Rao sketch requires axes=(ao,)");
    if (!axis_weights.empty() && axis_weights.size() != axes.size())
        throw std::invalid_argument("axis weights must match axes");
    for (double weight : axis_weights)
        if (!std::isfinite(weight) || weight <= 0)
            throw std::invalid_argument("axis weights must be finite and positive");
    std::vector<double> scales;
    if (axes.size() > 1 && algorithm != LocalAnchorSelectionAlgorithm::CentroidalVoronoi) {
        if (points.size() != weights.size())
            throw std::invalid_argument("points and weights must have equal lengths");
        const auto spec = axis_spec(axes, mo_coeff, nocc);
        std::vector<double> norms(axes.size(), 0.0);
        // Global (not per-batch) Frobenius norms over eligible grid points.
        // A bounded host pass also keeps online QR's streaming memory bound.
        std::vector<std::array<double, 3>> batch;
        batch.reserve(128);
        bool any_eligible = false;
        auto accumulate = [&]() {
            if (batch.empty()) return;
            const auto evaluated = evaluate_axis<Kokkos::DefaultHostExecutionSpace>(
                batch, axes, mo_coeff, nocc);
            auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, evaluated.values);
            for (std::size_t g = 0; g < batch.size(); ++g) {
                std::size_t offset = 0;
                for (std::size_t a = 0; a < axes.size(); ++a) {
                    const std::size_t width = axes[a] == AxisType::OccupiedOrbital ? nocc : nao();
                    for (std::size_t j = 0; j < width; ++j)
                        norms[a] = std::hypot(norms[a], values(g*spec.width + offset + j));
                    offset += width;
                }
            }
            batch.clear();
        };
        for (std::size_t g = 0; g < points.size(); ++g) {
            if (!std::isfinite(weights[g])) throw std::invalid_argument("grid weights must be finite");
            if (weights[g] == 0) continue;
            any_eligible = true;
            batch.push_back(points[g]);
            if (batch.size() == 128) accumulate();
        }
        accumulate();
        scales.reserve(axes.size());
        for (std::size_t a = 0; any_eligible && a < axes.size(); ++a) {
            if (!(norms[a] > 0) || !std::isfinite(norms[a]))
                throw std::invalid_argument("axis Frobenius norm must be finite and positive");
            const double scale = (axis_weights.empty() ? 1.0 : axis_weights[a]) / norms[a];
            if (!(scale > 0) || !std::isfinite(scale))
                throw std::invalid_argument("normalized axis scale must be finite and positive");
            scales.push_back(scale);
        }
    }
    switch (algorithm) {
    case LocalAnchorSelectionAlgorithm::QRColumnPivoting:
        return QRCPAnchorSelection::select(*this, points, weights, axes, tolerance,
                                           max_rank, mo_coeff, nocc, space, scales);
    case LocalAnchorSelectionAlgorithm::OnlineQRColumnPivoting:
        return OnlineQRAnchorSelection::select(*this, points, weights, axes, tolerance,
                                               max_rank, mo_coeff, nocc, space, batch_size, debug, scales);
    case LocalAnchorSelectionAlgorithm::RandomizedFeatureKhatriRao:
    case LocalAnchorSelectionAlgorithm::RandomizedAOQuadratureKhatriRao:
    case LocalAnchorSelectionAlgorithm::RandomizedAODensityKhatriRao:
    case LocalAnchorSelectionAlgorithm::RandomizedAODensityNormalizedKhatriRao:
        return RandomizedIDAnchorSelection::select(*this, points, weights, axes, tolerance,
            max_rank, mo_coeff, nocc, space, batch_size, debug,
            sketch_size, oversampling, seed, audit,
            algorithm == LocalAnchorSelectionAlgorithm::RandomizedAODensityNormalizedKhatriRao
                ? SketchPolicy::AODensityNormalized
                : algorithm == LocalAnchorSelectionAlgorithm::RandomizedAODensityKhatriRao
                ? SketchPolicy::AODensity
                : algorithm == LocalAnchorSelectionAlgorithm::RandomizedAOQuadratureKhatriRao
                ? SketchPolicy::AOQuadrature : SketchPolicy::FeatureQuadrature,
            scales, point_importance, ao_screening_threshold);
    case LocalAnchorSelectionAlgorithm::CentroidalVoronoi:
        return CVTAnchorSelection::select(points, weights, cvt_weights, max_rank,
                                          space, seed, cvt_max_iter, cvt_tol, debug);
    default:
        throw std::invalid_argument("unknown anchor selection algorithm");
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
