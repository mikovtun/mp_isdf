#pragma once

// libcint's generated header lacks C++ linkage guards.
extern "C" {
#include <cint.h>
}

#include <array>
#include <vector>

namespace isdf {

// Row-major values: result[point * nao + ao]. Inputs use libcint's ATM/BAS/ENV
// layout (e.g. PySCF mol._atm, mol._bas, mol._env). Supports real spherical
// contracted Gaussian AOs, including multiple contractions per shell.
std::vector<double> eval_ao(const std::vector<FINT>& atm,
                            const std::vector<FINT>& bas,
                            const std::vector<double>& env,
                            const std::vector<std::array<double, 3>>& points);

} // namespace isdf
