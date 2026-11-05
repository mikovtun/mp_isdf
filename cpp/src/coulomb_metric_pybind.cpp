#include "coulomb_metric.hpp"
#include "derivative_metric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void bind_coulomb_metric(py::module_& m)
{
    m.def("cholesky_eri", [](const isdf::LibcintContext& ctx, double tolerance,
                              std::size_t max_rank, const std::string& algorithm,
                              bool debug) {
        if (algorithm != "unblocked_greedy") throw std::invalid_argument("unknown Cholesky algorithm");
        isdf::ERICholesky result;
        { py::gil_scoped_release release; result = isdf::cholesky_eri(ctx,tolerance,max_rank,debug); }
        const auto rank = result.pivots.size();
        py::array_t<double> factor({static_cast<py::ssize_t>(rank),static_cast<py::ssize_t>(result.npair)});
        py::array_t<double> residual(static_cast<py::ssize_t>(result.npair));
        py::array_t<py::ssize_t> pivots(static_cast<py::ssize_t>(rank));
        std::copy(result.factors.begin(),result.factors.end(),factor.mutable_data());
        std::copy(result.residual.begin(),result.residual.end(),residual.mutable_data());
        for (std::size_t k=0;k<rank;++k) pivots.mutable_data()[k] = static_cast<py::ssize_t>(result.pivots[k]);
        return py::make_tuple(factor,pivots,residual);
    }, py::arg("context"), py::arg("tolerance")=1e-12,
       py::arg("max_rank")=0, py::arg("algorithm")="unblocked_greedy",
       py::arg("debug")=false,
       "Return (factors[rank,npair], zero-based pivots, residual diagonal) in packed AO-pair order.");
    m.def("derivative_cross_metric", [](const isdf::LibcintContext& ctx,
                                        const FloatArray& left, std::array<int,2> left_basis,
                                        const FloatArray& right, std::array<int,2> right_basis,
                                        double tolerance, std::size_t max_rank,
                                        bool debug) {
        const auto npair = ctx.nao()*ctx.nao();
        if (left.ndim()!=2 || right.ndim()!=2 ||
            static_cast<std::size_t>(left.shape(1))!=npair ||
            static_cast<std::size_t>(right.shape(1))!=npair)
            throw std::invalid_argument("expected coefficient arrays [nvec,nao*nao]");
        std::vector<double> a(left.data(),left.data()+left.size());
        std::vector<double> b(right.data(),right.data()+right.size());
        std::vector<double> values;
        { py::gil_scoped_release release;
          values = isdf::derivative_cross_metric(ctx,a,static_cast<std::size_t>(left.shape(0)),
                     left_basis,b,static_cast<std::size_t>(right.shape(0)),right_basis,
                     tolerance,max_rank,debug); }
        py::array_t<double> output({left.shape(0),right.shape(0)});
        std::copy(values.begin(),values.end(),output.mutable_data());
        return output;
    },py::arg("context"),py::arg("left_coeff"),py::arg("left_basis"),
      py::arg("right_coeff"),py::arg("right_basis"),py::arg("tolerance")=1e-12,
      py::arg("max_rank")=0,py::arg("debug")=false,
      "Integral-direct joint unblocked greedy Cholesky metric for two ordered AO-pair channels.");
    m.def("coulomb_metric", [](const isdf::LibcintContext& ctx, const FloatArray& transform,
                                const std::string& algorithm, double tolerance,
                                std::size_t max_rank, std::size_t batch_size,
                                bool debug) {
        if (algorithm != "unblocked_greedy") throw std::invalid_argument("unknown Coulomb metric algorithm");
        const auto nao = ctx.nao();
        if (transform.ndim()!=2 || static_cast<std::size_t>(transform.shape(1))!=nao*nao)
            throw std::invalid_argument("expected transform[nvec,nao*nao]");
        std::vector<double> coeff(transform.data(),transform.data()+transform.size());
        std::vector<double> values;
        { py::gil_scoped_release release;
          values = isdf::coulomb_metric(ctx,coeff,static_cast<std::size_t>(transform.shape(0)),
                                        tolerance,max_rank,batch_size,debug); }
        py::array_t<double> result({transform.shape(0),transform.shape(0)});
        std::copy(values.begin(),values.end(),result.mutable_data());
        return result;
    }, py::arg("context"), py::arg("transform"),py::arg("algorithm")="unblocked_greedy",
       py::arg("tolerance")=1e-12,py::arg("max_rank")=0,py::arg("batch_size")=64,
       py::arg("debug")=false,
       "AO-product Coulomb metric; transform uses ordered AO-pair columns. No anchor weights applied.");
}
