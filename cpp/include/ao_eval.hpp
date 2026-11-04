#pragma once

#include "libcint_wrapper.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace isdf {

// Reusable, owning libcint context. Construct once from PySCF tables; future
// routines can accept const LibcintContext& and use tables() without copying.
// Its AO shell metadata, transforms, and ENV are cached on the Kokkos device.
// Initialize Kokkos before construction and keep it alive through destruction.
class LibcintContext {
public:
    LibcintContext(std::vector<FINT> atm, std::vector<FINT> bas,
                   std::vector<double> env);
    ~LibcintContext();
    LibcintContext(const LibcintContext&) = delete;
    LibcintContext& operator=(const LibcintContext&) = delete;
    LibcintContext(LibcintContext&&) = delete;
    LibcintContext& operator=(LibcintContext&&) = delete;

    const LibcintWrapper& tables() const noexcept { return tables_; }
    std::size_t nao() const noexcept;
    std::vector<double> eval_ao(const std::vector<std::array<double, 3>>& points) const;
    std::vector<double> eval_ao_gradient(
        const std::vector<std::array<double, 3>>& points) const;

    // Low-level dispatcher; prefer eval_ao / eval_ao_gradient above.
    std::vector<double> evaluate(const std::vector<std::array<double, 3>>& points,
                                 bool gradient) const;

private:
    struct Impl;
    LibcintWrapper tables_;
    std::unique_ptr<Impl> impl_;
};

// Stateless convenience functions; for repeated calls prefer LibcintContext.
// Row-major values: result[point * nao + ao]. Inputs use libcint's ATM/BAS/ENV
// layout (e.g. PySCF mol._atm, mol._bas, mol._env). Supports real spherical
// contracted Gaussian AOs, including multiple contractions per shell.
std::vector<double> eval_ao(const std::vector<FINT>& atm,
                            const std::vector<FINT>& bas,
                            const std::vector<double>& env,
                            const std::vector<std::array<double, 3>>& points);

// Cartesian gradients of real spherical AOs, with layout
// gradient[(axis * npoints + point) * nao + ao], axis = x, y, z.
// Matches PySCF's GTOval_ip_sph convention (derivatives w.r.t. grid position).
std::vector<double> eval_ao_gradient(
    const std::vector<FINT>& atm,
    const std::vector<FINT>& bas,
    const std::vector<double>& env,
    const std::vector<std::array<double, 3>>& points);

} // namespace isdf
