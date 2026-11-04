#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace isdf {

template <class ExecutionSpace>
auto AnchorSelection::evaluate_axis(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<AxisType>& axes,
    const std::vector<double>& mo_coeff,
    std::size_t nocc) const -> AxisEvaluation<ExecutionSpace>
{
    const auto spec = axis_spec(axes, mo_coeff, nocc);
    if (points.size() > std::numeric_limits<std::size_t>::max() / spec.width) {
        throw std::overflow_error("axis evaluation is too large");
    }
    for (const auto& point : points) {
        for (double x : point) {
            if (!std::isfinite(x)) throw std::invalid_argument("grid coordinates must be finite");
        }
    }
    const std::size_t ng = points.size();
    const std::size_t nao = context_.nao();
    Kokkos::View<double*, ExecutionSpace> values("axis_features", ng * spec.width);
    if (ng == 0) return {spec.width, values};

    // Evaluate raw orbitals and assemble the axis in the same space; only
    // the optional MO coefficient matrix is staged from host storage.
    const auto d_ao = spec.need_ao
        ? context_.evaluate_in_space<ExecutionSpace>(points, false)
        : Kokkos::View<double*, ExecutionSpace>{};
    const auto d_grad = spec.need_grad
        ? context_.evaluate_in_space<ExecutionSpace>(points, true)
        : Kokkos::View<double*, ExecutionSpace>{};
    auto upload = [](const std::vector<double>& source, const char* label) {
        Kokkos::View<double*, ExecutionSpace> target(label, source.size());
        auto mirror = Kokkos::create_mirror_view(target);
        for (std::size_t i = 0; i < source.size(); ++i) mirror(i) = source[i];
        Kokkos::deep_copy(target, mirror);
        return target;
    };
    const auto d_coeff = std::find(axes.begin(), axes.end(), AxisType::OccupiedOrbital) != axes.end()
                       ? upload(mo_coeff, "axis_mo_coeff")
                       : Kokkos::View<double*, ExecutionSpace>{};
    std::size_t column = 0;
    for (AxisType axis : axes) {
        const std::size_t offset = column;
        const std::size_t width = axis == AxisType::OccupiedOrbital ? nocc : nao;
        const std::size_t stride = spec.width;
        const std::size_t component = axis == AxisType::AtomicOrbitalGradX ? 0
                                    : axis == AxisType::AtomicOrbitalGradY ? 1 : 2;
        Kokkos::parallel_for("isdf_assemble_axis", Kokkos::RangePolicy<ExecutionSpace>(0, ng * width),
                             KOKKOS_LAMBDA(const std::size_t index) {
            const std::size_t g = index / width;
            const std::size_t j = index % width;
            double value = 0.0;
            if (axis == AxisType::AtomicOrbital) {
                value = d_ao(g * nao + j);
            } else if (axis == AxisType::OccupiedOrbital) {
                for (std::size_t k = 0; k < nao; ++k) {
                    value += d_ao(g * nao + k) * d_coeff(k * nao + j);
                }
            } else {
                value = d_grad((component * ng + g) * nao + j);
            }
            values(g * stride + offset + j) = value;
        });
        column += width;
    }
    return {spec.width, values};
}

} // namespace isdf
