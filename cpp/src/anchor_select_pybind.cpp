#include "anchor_select.hpp"
#include "anchor_select_rbrp.hpp"
#include "anchor_select_rbrp_device.hpp"
#include "evaluation_space_pybind.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <stdexcept>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace {
using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<std::array<double, 3>> read_points(const FloatArray& points)
{
    if (points.ndim() != 2 || points.shape(1) != 3)
        throw std::invalid_argument("expected points[ngrid,3]");
    std::vector<std::array<double, 3>> grid(static_cast<std::size_t>(points.shape(0)));
    auto xyz = points.unchecked<2>();
    for (py::ssize_t i = 0; i < points.shape(0); ++i)
        for (int j = 0; j < 3; ++j) grid[i][j] = xyz(i, j);
    return grid;
}

std::vector<AxisType> read_axes(py::object axes)
{
    if (py::isinstance<py::list>(axes) || py::isinstance<py::tuple>(axes))
        return axes.cast<std::vector<AxisType>>();
    return {axes.cast<AxisType>()};
}

std::vector<double> read_coefficients(const isdf::AnchorSelection& selector, py::object mo_coeff)
{
    if (mo_coeff.is_none()) return {};
    const auto c = mo_coeff.cast<FloatArray>();
    const auto nao = static_cast<py::ssize_t>(selector.nao());
    if (c.ndim() != 2 || c.shape(0) != nao || c.shape(1) != nao)
        throw std::invalid_argument("mo_coeff must have shape (nao,nao)");
    return {c.data(), c.data() + c.size()};
}

py::array_t<double> evaluate_axis_array(const isdf::AnchorSelection& selector,
                                        const std::vector<std::array<double, 3>>& points,
                                        const std::vector<AxisType>& axes,
                                        const std::vector<double>& coefficients,
                                        std::size_t nocc, EvaluationSpace space)
{
    return std::visit([&](const auto& axis) {
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, axis.values);
        py::array_t<double> result({static_cast<py::ssize_t>(points.size()),
                                    static_cast<py::ssize_t>(axis.width)});
        std::copy_n(host.data(), host.size(), result.mutable_data());
        return result;
    }, selector.evaluate_axis(points, axes, coefficients, nocc, space));
}

py::dict rbrp_to_python(const isdf::RBRPResult& result, py::ssize_t n,
                        py::ssize_t width, bool compute_id)
{
    const auto rank = static_cast<py::ssize_t>(result.indices.size());
    py::dict output;
    output["indices"] = py::cast(result.indices);
    output["relative_residual"] = result.relative_residual;
    output["status"] = result.status;
    output["condition"] = result.condition;
    output["retained_rank"] = result.retained_rank;
    py::array_t<double> history({static_cast<py::ssize_t>(result.history.size()/4), py::ssize_t(4)});
    std::copy(result.history.begin(), result.history.end(), history.mutable_data());
    output["history"] = history;
    if (compute_id) {
        py::array_t<double> theta({rank, n});
        std::copy(result.interp_vecs.begin(), result.interp_vecs.end(), theta.mutable_data());
        py::array_t<double> transform({rank, width*width});
        std::copy(result.transform.begin(), result.transform.end(), transform.mutable_data());
        output["interp_vecs"] = theta;
        output["transform"] = transform;
    }
    return output;
}
} // namespace

void bind_anchor_select(py::module_& m)
{
    m.def("rbrp_ao_cpu", [](const FloatArray& ao, const FloatArray& weights,
                             std::size_t batch_size, std::size_t block_size,
                             double tolerance, std::size_t max_rank, std::uint64_t seed,
                             py::object tau_block, double rank_rtol,
                             std::size_t refresh_every, double svd_rtol, bool compute_id) {
        if (ao.ndim() != 2 || weights.ndim() != 1 || weights.shape(0) != ao.shape(0))
            throw std::invalid_argument("expected ao[grid,nao] and weights[grid]");
        const auto tb = tau_block.is_none() ? std::numeric_limits<double>::quiet_NaN()
                                             : tau_block.cast<double>();
        auto result = isdf::rbrp_ao_cpu(ao.data(), weights.data(),
            static_cast<std::size_t>(ao.shape(0)), static_cast<std::size_t>(ao.shape(1)),
            batch_size, block_size, tolerance, max_rank, seed,
            tb, rank_rtol, refresh_every, svd_rtol, compute_id);
        return rbrp_to_python(result, ao.shape(0), ao.shape(1), compute_id);
    }, py::arg("ao"), py::arg("weights"), py::arg("batch_size") = 128,
       py::arg("block_size") = 16, py::arg("tolerance") = 1e-10,
       py::arg("max_rank") = 0, py::arg("seed") = 7,
       py::arg("tau_block") = py::none(), py::arg("rank_rtol") = 1e-12,
       py::arg("refresh_every") = 8, py::arg("svd_rtol") = 1e-12,
       py::arg("compute_id") = false,
       "Host AO-only RBRP on nonzero grid rows; optional Algorithm 5.1 ID.");
#ifdef KOKKOS_ENABLE_CUDA
    m.def("rbrp_ao_device", [](const isdf::AnchorSelection& selector,
        const FloatArray& points, const FloatArray& weights,
        std::size_t batch_size, std::size_t block_size, double tolerance,
        std::size_t max_rank, std::uint64_t seed, py::object tau_block,
        double rank_rtol, std::size_t refresh_every, double svd_rtol,
        bool compute_id) {
        if (weights.ndim()!=1 || weights.shape(0)!=points.shape(0))
            throw std::invalid_argument("expected weights[ngrid]");
        auto grid=read_points(points);
        const auto tb=tau_block.is_none() ? std::numeric_limits<double>::quiet_NaN()
                                         : tau_block.cast<double>();
        auto result=isdf::rbrp_ao_device(selector,grid,
            std::vector<double>(weights.data(),weights.data()+weights.size()),
            batch_size,block_size,tolerance,max_rank,seed,tb,
            rank_rtol,refresh_every,svd_rtol,compute_id);
        return rbrp_to_python(result,points.shape(0),
            static_cast<py::ssize_t>(selector.nao()),compute_id);
    }, py::arg("selector"), py::arg("points"), py::arg("weights"),
       py::arg("batch_size")=128, py::arg("block_size")=16,
       py::arg("tolerance")=1e-10, py::arg("max_rank")=0,
       py::arg("seed")=7, py::arg("tau_block")=py::none(),
       py::arg("rank_rtol")=1e-12, py::arg("refresh_every")=8,
       py::arg("svd_rtol")=1e-12, py::arg("compute_id")=false,
       "CUDA AO-only RBRP; evaluates AOs in device batches and returns optional ID.");
#endif
    py::enum_<AxisType>(m, "AxisType")
        .value("AtomicOrbital", AxisType::AtomicOrbital)
        .value("AtomicOrbitalGradX", AxisType::AtomicOrbitalGradX)
        .value("AtomicOrbitalGradY", AxisType::AtomicOrbitalGradY)
        .value("AtomicOrbitalGradZ", AxisType::AtomicOrbitalGradZ)
        .value("OccupiedOrbital", AxisType::OccupiedOrbital);
    py::enum_<LocalAnchorSelectionAlgorithm>(m, "LocalAnchorSelectionAlgorithm")
        .value("QRColumnPivoting", LocalAnchorSelectionAlgorithm::QRColumnPivoting)
        .value("OnlineQRColumnPivoting", LocalAnchorSelectionAlgorithm::OnlineQRColumnPivoting)
        .value("RandomizedFeatureKhatriRao", LocalAnchorSelectionAlgorithm::RandomizedFeatureKhatriRao)
        .value("RandomizedAOQuadratureKhatriRao", LocalAnchorSelectionAlgorithm::RandomizedAOQuadratureKhatriRao)
        .value("CentroidalVoronoi", LocalAnchorSelectionAlgorithm::CentroidalVoronoi);
    py::enum_<DistributedAnchorSelectionAlgorithm>(m, "DistributedAnchorSelectionAlgorithm")
        .value("SVD", DistributedAnchorSelectionAlgorithm::SVD)
        .value("RandomizedQRColumnPivoting", DistributedAnchorSelectionAlgorithm::RandomizedQRColumnPivoting);

    py::class_<isdf::AnchorSelection>(m, "AnchorSelection")
        .def(py::init<const isdf::LibcintContext&>(), py::arg("context"),
             py::keep_alive<1, 2>())
        .def("select", [](const isdf::AnchorSelection& selector,
                          const FloatArray& points, const FloatArray& weights,
                          py::object axes, double tolerance, std::size_t max_rank,
                          LocalAnchorSelectionAlgorithm algorithm,
                          py::object mo_coeff, std::size_t nocc,
                          const std::string& space, std::size_t batch_size, bool debug,
                          std::size_t sketch_size, std::size_t oversampling,
                          std::uint64_t seed, bool audit, py::object cvt_weights,
                          std::size_t cvt_max_iter, double cvt_tol) {
            auto grid = read_points(points);
            if (weights.ndim() != 1 || weights.shape(0) != points.shape(0))
                throw std::invalid_argument("expected weights[ngrid]");
            auto axis_list = read_axes(axes);
            auto coefficients = read_coefficients(selector, mo_coeff);
            std::vector<double> w(weights.data(), weights.data() + weights.size());
            std::vector<double> density;
            if (!cvt_weights.is_none()) {
                auto input = cvt_weights.cast<FloatArray>();
                if (input.ndim() != 1 || input.shape(0) != points.shape(0))
                    throw std::invalid_argument("cvt_weights must have shape (ngrid,)");
                density.assign(input.data(), input.data() + input.size());
            }
            return selector.select(grid, w, axis_list, tolerance, max_rank,
                                   algorithm, coefficients, nocc,
                                   parse_evaluation_space(space), batch_size, debug,
                                   sketch_size, oversampling, seed, audit,
                                   density, cvt_max_iter, cvt_tol);
        }, py::arg("points"), py::arg("weights"), py::arg("axes"),
           py::arg("tolerance") = 1e-10, py::arg("max_rank") = 0,
           py::arg("algorithm") = LocalAnchorSelectionAlgorithm::QRColumnPivoting,
           py::arg("mo_coeff") = py::none(), py::arg("nocc") = 0,
           py::arg("space") = "auto", py::arg("batch_size") = 128,
           py::arg("debug") = false,
           py::arg("sketch_size") = 128, py::arg("oversampling") = 16,
           py::arg("seed") = 7, py::arg("audit") = false,
           py::arg("cvt_weights") = py::none(), py::arg("cvt_max_iter") = 100,
           py::arg("cvt_tol") = 1e-5,
           "Select original grid indices: QRCP returns pivot order; CVT returns sorted, distinct grid sites. CVT requires max_rank and cvt_weights (density or absolute quadrature weights).")
        .def("evaluate_axis", [](const isdf::AnchorSelection& selector,
                                  const FloatArray& points, py::object axes,
                                  py::object mo_coeff, std::size_t nocc,
                                  const std::string& space) {
            auto grid = read_points(points);
            auto axis_list = read_axes(axes);
            auto coefficients = read_coefficients(selector, mo_coeff);
            return evaluate_axis_array(selector, grid, axis_list, coefficients, nocc,
                                       parse_evaluation_space(space));
        }, py::arg("points"), py::arg("axes"), py::arg("mo_coeff") = py::none(),
           py::arg("nocc") = 0, py::arg("space") = "auto",
           "Evaluate concatenated axis functions in the requested Kokkos execution space.");

    py::class_<isdf::DistributedAnchorSelection>(m, "DistributedAnchorSelection")
        .def(py::init<const isdf::LibcintContext&>(), py::arg("context"),
             py::keep_alive<1, 2>())
        .def("select", [](const isdf::DistributedAnchorSelection&,
                          py::object, py::object, py::object,
                          DistributedAnchorSelectionAlgorithm) {
            throw std::logic_error("distributed SLATE SVD anchor selection is not implemented");
        }, py::arg("points"), py::arg("weights"), py::arg("axes"),
           py::arg("algorithm") = DistributedAnchorSelectionAlgorithm::SVD);
}
