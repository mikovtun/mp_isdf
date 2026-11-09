#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace isdf::detail {
// Factor an m-by-n column-major sketch/matrix in place; 1-based pivots.
std::vector<std::int64_t> qrcp_cpu(double* matrix, std::size_t m, std::size_t n);
#ifdef ISDF_ENABLE_CUDA
std::vector<std::int64_t> qrcp_magma(double* device_matrix, std::size_t m, std::size_t n);
#endif
} // namespace isdf::detail
