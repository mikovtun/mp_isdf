#pragma once

#include "enums.hpp"
#include "libcint_wrapper.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>
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
    std::vector<double> eval_ao(const std::vector<std::array<double, 3>>& points,
                                EvaluationSpace space = EvaluationSpace::Auto) const;
    std::vector<double> eval_ao_gradient(
        const std::vector<std::array<double, 3>>& points,
        EvaluationSpace space = EvaluationSpace::Auto) const;

    // Low-level dispatcher; prefer eval_ao / eval_ao_gradient above.
    std::vector<double> evaluate(const std::vector<std::array<double, 3>>& points,
                                 bool gradient, EvaluationSpace space = EvaluationSpace::Auto) const;

    // Raw AO (or gradient) values remain in the requested execution space.
    // Supported spaces are the default host space, and CUDA when enabled.
    // The configured default execution space also remains available.
    template <class ExecutionSpace>
    Kokkos::View<double*, ExecutionSpace> evaluate_in_space(
        const std::vector<std::array<double, 3>>& points, bool gradient) const
    {
        if constexpr (std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
            return evaluate_host_view(points, gradient);
#ifdef KOKKOS_ENABLE_CUDA
        } else if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
            return evaluate_cuda_view(points, gradient);
#endif
        } else {
            static_assert(std::is_same_v<ExecutionSpace, Kokkos::DefaultExecutionSpace>,
                          "AO evaluation supports the host or configured device execution space");
            return evaluate_default_view(points, gradient);
        }
    }

    // Public because CUDA extended lambdas cannot reside in private methods.
    // Prefer evaluate_in_space; evaluate_impl is only instantiated internally.
    template <class ExecutionSpace>
    Kokkos::View<double*, ExecutionSpace> evaluate_impl(
        const std::vector<std::array<double, 3>>& points, bool gradient) const;

private:
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> evaluate_default_view(
        const std::vector<std::array<double, 3>>& points, bool gradient) const;
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> evaluate_host_view(
        const std::vector<std::array<double, 3>>& points, bool gradient) const;
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::View<double*, Kokkos::Cuda> evaluate_cuda_view(
        const std::vector<std::array<double, 3>>& points, bool gradient) const;
#endif
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
                            const std::vector<std::array<double, 3>>& points,
                            EvaluationSpace space = EvaluationSpace::Auto);

// Cartesian gradients of real spherical AOs, with layout
// gradient[(axis * npoints + point) * nao + ao], axis = x, y, z.
// Matches PySCF's GTOval_ip_sph convention (derivatives w.r.t. grid position).
std::vector<double> eval_ao_gradient(
    const std::vector<FINT>& atm,
    const std::vector<FINT>& bas,
    const std::vector<double>& env,
    const std::vector<std::array<double, 3>>& points,
    EvaluationSpace space = EvaluationSpace::Auto);

} // namespace isdf
