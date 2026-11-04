#pragma once

#include "enums.hpp"

#include <stdexcept>
#include <string>

inline EvaluationSpace parse_evaluation_space(const std::string& space)
{
    if (space == "auto" || space == "default") return EvaluationSpace::Auto;
    if (space == "host") return EvaluationSpace::Host;
    if (space == "device") return EvaluationSpace::Device;
    throw std::invalid_argument("space must be 'auto', 'host', or 'device'");
}
