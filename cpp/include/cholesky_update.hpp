#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace isdf {

// Largest remaining residual, with the first pair winning exact ties. The
// eligibility pointer is optional: future pivot screening can exclude rows
// from selection without dropping their factor entries (needed for projection).
struct CholeskyCandidate {
    double residual;
    std::size_t index;
    bool invalid;
};

struct CholeskyCorrection {
    using value_type = CholeskyCandidate;
    const double* integrals;
    const double* old_factors;
    const double* pivot_factors;
    double* new_factors;
    double* residual;
    const unsigned char* eligible;
    std::size_t dimension, rank, pivot;
    double denominator;

    KOKKOS_INLINE_FUNCTION void init(value_type& candidate) const {
        candidate = {-1.0, dimension, false}; // clamped residuals are nonnegative
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const {
        const bool invalid = dst.invalid || src.invalid;
        if (src.residual > dst.residual ||
            (src.residual == dst.residual && src.index < dst.index)) {
            dst.residual = src.residual;
            dst.index = src.index;
        }
        dst.invalid = invalid;
    }

    // One writer per pair; all previous factor columns and pivot values are
    // immutable for the duration of the reduction. Preserve the old k order.
    void operator()(std::size_t p, value_type& best) const {
        double value = integrals[p];
        for (std::size_t k = 0; k < rank; ++k)
            value -= old_factors[k*dimension+p] * pivot_factors[k];
        value /= denominator;
        new_factors[p] = value;
        if (!std::isfinite(value)) {
            best.invalid = true;
            return;
        }
        residual[p] = p == pivot ? 0.0 : std::max(0.0, residual[p] - value*value);
        if ((!eligible || eligible[p]) &&
            (residual[p] > best.residual ||
             (residual[p] == best.residual && p < best.index))) {
            best.residual = residual[p];
            best.index = p;
        }
    }
};

// Append the current factor column and return the *next* pivot candidate.
// This keeps factor storage dense even if an eligibility mask is added later.
inline CholeskyCandidate correct_cholesky_column(
    const std::vector<double>& integrals, std::vector<double>& factors,
    std::vector<double>& residual, std::size_t pivot, double pivot_residual,
    const char* overflow_message, const char* nonfinite_message,
    const unsigned char* eligible = nullptr)
{
    const auto dimension = residual.size();
    const auto rank = factors.size()/dimension;
    if (dimension > std::numeric_limits<std::size_t>::max()/(rank+1))
        throw std::overflow_error(overflow_message);
    std::vector<double> pivot_factors(rank);
    for (std::size_t k = 0; k < rank; ++k)
        pivot_factors[k] = factors[k*dimension+pivot];
    factors.resize((rank+1)*dimension);
    CholeskyCorrection correction{integrals.data(), factors.data(), pivot_factors.data(),
                                  factors.data()+rank*dimension, residual.data(), eligible,
                                  dimension, rank, pivot, std::sqrt(pivot_residual)};
    CholeskyCandidate next;
    // Avoid launching a host team for a trivially small pair set.
    if (dimension < 64) {
        correction.init(next);
        for (std::size_t p = 0; p < dimension; ++p) correction(p, next);
    } else {
        Kokkos::DefaultHostExecutionSpace exec;
        Kokkos::parallel_reduce("isdf_cholesky_correct_and_select",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(exec, 0, dimension),
            correction, next);
        exec.fence();
    }
    if (next.invalid) throw std::runtime_error(nonfinite_message);
    return next;
}

} // namespace isdf
