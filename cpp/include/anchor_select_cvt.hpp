#pragma once

#include "enums.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace isdf {

class CVTAnchorSelection {
public:
    static std::vector<std::size_t> select(
        const std::vector<std::array<double, 3>>& points,
        const std::vector<double>& weights, const std::vector<double>& importance,
        std::size_t rank, EvaluationSpace space, std::uint64_t seed,
        std::size_t max_iter, double tol, bool debug);
};

} // namespace isdf
