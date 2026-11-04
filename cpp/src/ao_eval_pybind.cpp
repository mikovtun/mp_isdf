#include "ao_eval.hpp"
#include "evaluation_space_pybind.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {
using IntArray = py::array_t<FINT, py::array::c_style | py::array::forcecast>;
using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<std::array<double, 3>> read_points(const FloatArray& coords)
{
    if (coords.ndim() != 2 || coords.shape(1) != 3) {
        throw std::invalid_argument("expected coords[ngrid,3] in Bohr");
    }
    std::vector<std::array<double, 3>> points(static_cast<std::size_t>(coords.shape(0)));
    auto xyz = coords.unchecked<2>();
    for (py::ssize_t p = 0; p < coords.shape(0); ++p) {
        for (int k = 0; k < 3; ++k) points[p][k] = xyz(p, k);
    }
    return points;
}

std::unique_ptr<isdf::LibcintContext> make_context(const IntArray& atm,
                                                   const IntArray& bas,
                                                   const FloatArray& env)
{
    if (atm.ndim() != 2 || atm.shape(1) != ATM_SLOTS ||
        bas.ndim() != 2 || bas.shape(1) != BAS_SLOTS || env.ndim() != 1) {
        throw std::invalid_argument("expected atm[natm,6], bas[nbas,8], env[nenv]");
    }
    std::vector<FINT> atoms(atm.data(), atm.data() + atm.size());
    std::vector<FINT> basis(bas.data(), bas.data() + bas.size());
    std::vector<double> environment(env.data(), env.data() + env.size());
    return std::make_unique<isdf::LibcintContext>(
        std::move(atoms), std::move(basis), std::move(environment));
}

py::array_t<double> evaluate(const isdf::LibcintContext& context,
                             const FloatArray& coords, bool gradient,
                             EvaluationSpace space)
{
    auto points = read_points(coords);
    std::vector<double> values;
    {
        py::gil_scoped_release release;
        values = gradient ? context.eval_ao_gradient(points, space) : context.eval_ao(points, space);
    }
    std::vector<py::ssize_t> shape = gradient
        ? std::vector<py::ssize_t>{3, coords.shape(0), static_cast<py::ssize_t>(context.nao())}
        : std::vector<py::ssize_t>{coords.shape(0), static_cast<py::ssize_t>(context.nao())};
    py::array_t<double> out(shape);
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}
} // namespace

void bind_anchor_select(py::module_& m);

PYBIND11_MODULE(isdf_ao, m)
{
    m.doc() = "Kokkos spherical AO values and gradients for libcint/PySCF tables";
#ifdef KOKKOS_ENABLE_CUDA
    m.attr("cuda_enabled") = true;
#else
    m.attr("cuda_enabled") = false;
#endif
    const bool initialized_here = !Kokkos::is_initialized();
    if (initialized_here) Kokkos::initialize();
    m.add_object("_kokkos_lifetime", py::capsule(new bool(initialized_here), [](void* p) {
        const bool owned = *static_cast<bool*>(p);
        delete static_cast<bool*>(p);
        if (owned && Kokkos::is_initialized() && !Kokkos::is_finalized()) {
            Kokkos::finalize();
        }
    }));

    py::class_<isdf::LibcintContext>(m, "LibcintContext",
        "Own a private copy of libcint tables and cache AO data on the Kokkos device.")
        .def(py::init(&make_context), py::arg("atm"), py::arg("bas"), py::arg("env"))
        .def_property_readonly("nao", &isdf::LibcintContext::nao)
        .def("eval_ao", [](const isdf::LibcintContext& ctx, const FloatArray& coords,
                            const std::string& space) {
            return evaluate(ctx, coords, false, parse_evaluation_space(space));
        }, py::arg("coords"), py::arg("space") = "auto",
           "Return real spherical AOs with shape (ngrid, nao).")
        .def("eval_ao_gradient", [](const isdf::LibcintContext& ctx, const FloatArray& coords,
                                     const std::string& space) {
            return evaluate(ctx, coords, true, parse_evaluation_space(space));
        }, py::arg("coords"), py::arg("space") = "auto",
           "Return x,y,z AO gradients with shape (3, ngrid, nao).");

    // Retain the original array-based interface for existing Python callers.
    m.def("eval_ao", [](const IntArray& atm, const IntArray& bas,
                        const FloatArray& env, const FloatArray& coords,
                        const std::string& space) {
        auto ctx = make_context(atm, bas, env);
        return evaluate(*ctx, coords, false, parse_evaluation_space(space));
    }, py::arg("atm"), py::arg("bas"), py::arg("env"), py::arg("coords"),
       py::arg("space") = "auto");
    m.def("eval_ao_gradient", [](const IntArray& atm, const IntArray& bas,
                                 const FloatArray& env, const FloatArray& coords,
                                 const std::string& space) {
        auto ctx = make_context(atm, bas, env);
        return evaluate(*ctx, coords, true, parse_evaluation_space(space));
    }, py::arg("atm"), py::arg("bas"), py::arg("env"), py::arg("coords"),
       py::arg("space") = "auto");
    bind_anchor_select(m);
}
