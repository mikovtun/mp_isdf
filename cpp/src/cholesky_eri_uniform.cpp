#include "cholesky_eri_impl.hpp"

namespace isdf {
ERICholesky cholesky_eri_uniform(const LibcintContext& ctx, double tolerance,
                                 std::size_t max_rank, std::uint64_t seed, bool debug)
{
    return cholesky_eri_impl(ctx, tolerance, max_rank, debug,
                             CholeskyAlgorithm::UnblockedUniform, seed);
}
} // namespace isdf
