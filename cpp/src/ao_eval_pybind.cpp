#include "ao_eval.hpp"

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(isdf_ao, m)
{
    m.doc() = "Kokkos spherical AO evaluation for libcint/PySCF tables";
    const bool initialized_here = !Kokkos::is_initialized();
    if (initialized_here) Kokkos::initialize();
    m.add_object("_kokkos_lifetime", py::capsule(new bool(initialized_here), [](void* p) {
        const bool owned = *static_cast<bool*>(p);
        delete static_cast<bool*>(p);
        if (owned && Kokkos::is_initialized() && !Kokkos::is_finalized()) {
            Kokkos::finalize();
        }
    }));

    m.def("eval_ao", [](py::array_t<FINT, py::array::c_style | py::array::forcecast> atm,
                         py::array_t<FINT, py::array::c_style | py::array::forcecast> bas,
                         py::array_t<double, py::array::c_style | py::array::forcecast> env,
                         py::array_t<double, py::array::c_style | py::array::forcecast> coords) {
        if (atm.ndim() != 2 || atm.shape(1) != ATM_SLOTS ||
            bas.ndim() != 2 || bas.shape(1) != BAS_SLOTS ||
            env.ndim() != 1 || coords.ndim() != 2 || coords.shape(1) != 3) {
            throw std::invalid_argument("expected atm[natm,6], bas[nbas,8], env[nenv], coords[ngrid,3]");
        }
        std::vector<FINT> atoms(atm.data(), atm.data() + atm.size());
        std::vector<FINT> basis(bas.data(), bas.data() + bas.size());
        std::vector<double> environment(env.data(), env.data() + env.size());
        std::vector<std::array<double, 3>> points(static_cast<std::size_t>(coords.shape(0)));
        auto xyz = coords.unchecked<2>();
        for (py::ssize_t p = 0; p < coords.shape(0); ++p) {
            for (int k = 0; k < 3; ++k) points[p][k] = xyz(p, k);
        }
        // Release the GIL while Kokkos evaluates the independent grid points.
        std::vector<double> values;
        {
            py::gil_scoped_release release;
            values = isdf::eval_ao(atoms, basis, environment, points);
        }
        const py::ssize_t nao = points.empty() ? [&] {
            py::ssize_t count = 0;
            for (std::size_t sh = 0; sh < basis.size() / BAS_SLOTS; ++sh)
                count += (2 * basis[sh * BAS_SLOTS + ANG_OF] + 1) * basis[sh * BAS_SLOTS + NCTR_OF];
            return count;
        }() : static_cast<py::ssize_t>(values.size() / points.size());
        py::array_t<double> out({coords.shape(0), nao});
        std::copy(values.begin(), values.end(), out.mutable_data());
        return out;
    }, py::arg("atm"), py::arg("bas"), py::arg("env"), py::arg("coords"),
       "Evaluate real spherical AOs at Cartesian points (Bohr), returning (ngrid, nao).");
}
