#pragma once
#include "anchor_select.hpp"
#include <cstdint>

namespace isdf {

// Two policies share the factored Gaussian Khatri-Rao sketch and sketch QRCP:
// general axis features or AO-only quadrature, both weighted by abs(weights).
// Pivots only: the packed sketch QR and its upper triangular R are the
// natural continuation point for future ID coefficients X = R11^{-1} R1:.
class RandomizedIDAnchorSelection {
public:
    static std::vector<std::size_t> select(
        const AnchorSelection& selector,
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& weights, const std::vector<AxisType>& axes,
        double tolerance, std::size_t max_rank,
        const std::vector<double>& mo_coeff, std::size_t nocc,
        EvaluationSpace space, std::size_t batch_size, bool debug,
        std::size_t sketch_size, std::size_t oversampling,
        std::uint64_t seed, bool audit, bool positive_ao_quadrature = false);

    // Public because CUDA extended lambdas require a public enclosing function.
    template <class Space>
    static std::vector<std::size_t> select_in_space(
        const AnchorSelection& selector,
        const std::vector<std::array<double, 3>>& kept_points,
        const std::vector<double>& kept_weights,
        const std::vector<std::size_t>& original,
        const std::vector<AxisType>& axes,
        double tolerance, std::size_t max_rank,
        const std::vector<double>& mo_coeff, std::size_t nocc,
        std::size_t batch_size, bool debug, std::size_t sketch_size,
        std::size_t oversampling, std::uint64_t seed, bool audit,
        bool positive_ao_quadrature);
};
} // namespace isdf
