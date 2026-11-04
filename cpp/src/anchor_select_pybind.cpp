#include "anchor_select.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

void bind_anchor_select(py::module_& m)
{
    py::enum_<AxisType>(m, "AxisType")
        .value("AtomicOrbital", AxisType::AtomicOrbital)
        .value("AtomicOrbitalGradX", AxisType::AtomicOrbitalGradX)
        .value("AtomicOrbitalGradY", AxisType::AtomicOrbitalGradY)
        .value("AtomicOrbitalGradZ", AxisType::AtomicOrbitalGradZ)
        .value("OccupiedOrbital", AxisType::OccupiedOrbital);
    py::enum_<LocalAnchorSelectionAlgorithm>(m, "LocalAnchorSelectionAlgorithm")
        .value("QRColumnPivoting", LocalAnchorSelectionAlgorithm::QRColumnPivoting)
        .value("RandomizedQRColumnPivoting", LocalAnchorSelectionAlgorithm::RandomizedQRColumnPivoting);
    py::enum_<DistributedAnchorSelectionAlgorithm>(m, "DistributedAnchorSelectionAlgorithm")
        .value("SVD", DistributedAnchorSelectionAlgorithm::SVD)
        .value("RandomizedQRColumnPivoting", DistributedAnchorSelectionAlgorithm::RandomizedQRColumnPivoting);

    using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
    py::class_<isdf::AnchorSelection>(m, "AnchorSelection")
        .def(py::init<const isdf::LibcintContext&>(), py::arg("context"),
             py::keep_alive<1, 2>())
        .def("select", [](const isdf::AnchorSelection& selector,
                          const FloatArray& points, const FloatArray& weights,
                          py::object axes, double tolerance, std::size_t max_rank,
                          LocalAnchorSelectionAlgorithm algorithm,
                          py::object mo_coeff, std::size_t nocc) {
            if (points.ndim() != 2 || points.shape(1) != 3 ||
                weights.ndim() != 1 || weights.shape(0) != points.shape(0)) {
                throw std::invalid_argument("expected points[ngrid,3] and weights[ngrid]");
            }
            std::vector<AxisType> axis_list;
            if (py::isinstance<py::list>(axes) || py::isinstance<py::tuple>(axes)) {
                axis_list = axes.cast<std::vector<AxisType>>();
            } else {
                axis_list = {axes.cast<AxisType>()};
            }
            std::vector<std::array<double, 3>> grid(static_cast<std::size_t>(points.shape(0)));
            auto xyz = points.unchecked<2>();
            for (py::ssize_t i = 0; i < points.shape(0); ++i)
                for (int j = 0; j < 3; ++j) grid[i][j] = xyz(i, j);
            std::vector<double> w(weights.data(), weights.data() + weights.size());
            std::vector<double> coefficients;
            if (!mo_coeff.is_none()) {
                const auto c = mo_coeff.cast<FloatArray>();
                const auto nao = static_cast<py::ssize_t>(selector.nao());
                if (c.ndim() != 2 || c.shape(0) != nao || c.shape(1) != nao)
                    throw std::invalid_argument("mo_coeff must have shape (nao,nao)");
                coefficients.assign(c.data(), c.data() + c.size());
            }
            return selector.select(grid, w, axis_list, tolerance, max_rank,
                                   algorithm, coefficients, nocc);
        }, py::arg("points"), py::arg("weights"), py::arg("axes"),
           py::arg("tolerance") = 1e-10, py::arg("max_rank") = 0,
           py::arg("algorithm") = LocalAnchorSelectionAlgorithm::QRColumnPivoting,
           py::arg("mo_coeff") = py::none(), py::arg("nocc") = 0,
           "Select original grid indices in pivot order (QRCP on weighted pair-product transpose).");

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
