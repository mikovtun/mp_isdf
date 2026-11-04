#include "coulomb_metric.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void bind_coulomb_metric(py::module_& m)
{
    m.def("cholesky_eri", [](const isdf::LibcintContext& ctx, double tolerance,
                              std::size_t max_rank, const std::string& algorithm) {
        if (algorithm != "unblocked_greedy") throw std::invalid_argument("unknown Cholesky algorithm");
        isdf::ERICholesky result;
        { py::gil_scoped_release release; result = isdf::cholesky_eri(ctx,tolerance,max_rank); }
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
       "Return (factors[rank,npair], zero-based pivots, residual diagonal) in packed AO-pair order.");
    m.def("coulomb_metric", [](const isdf::LibcintContext& ctx, const FloatArray& transform,
                                const std::string& algorithm, double tolerance,
                                std::size_t max_rank, std::size_t batch_size) {
        if (algorithm != "unblocked_greedy") throw std::invalid_argument("unknown Coulomb metric algorithm");
        const auto nao = ctx.nao();
        if (transform.ndim()!=2 || static_cast<std::size_t>(transform.shape(1))!=nao*nao)
            throw std::invalid_argument("expected transform[nvec,nao*nao]");
        std::vector<double> coeff(transform.data(),transform.data()+transform.size());
        std::vector<double> values;
        { py::gil_scoped_release release;
          values = isdf::coulomb_metric(ctx,coeff,static_cast<std::size_t>(transform.shape(0)),
                                        tolerance,max_rank,batch_size); }
        py::array_t<double> result({transform.shape(0),transform.shape(0)});
        std::copy(values.begin(),values.end(),result.mutable_data());
        return result;
    }, py::arg("context"), py::arg("transform"),py::arg("algorithm")="unblocked_greedy",
       py::arg("tolerance")=1e-12,py::arg("max_rank")=0,py::arg("batch_size")=64,
       "AO-product Coulomb metric; transform uses ordered AO-pair columns. No anchor weights applied.");
}
