#include "cholesky_eri_impl.hpp"
#include "parallel_shell_tasks.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace isdf {
namespace {
struct Shell { FINT id; std::size_t offset, size; };
std::size_t pair_index(std::size_t a, std::size_t b) { return a*(a+1)/2+b; }

std::vector<Shell> shells_for(const LibcintContext& ctx)
{
    std::vector<Shell> shells;
    std::size_t offset = 0;
    for (FINT s = 0; s < ctx.tables().nbas(); ++s) {
        const auto n = static_cast<std::size_t>(CINTcgto_spheric(s, ctx.tables().bas_data()));
        shells.push_back({s, offset, n});
        offset += n;
    }
    if (offset != ctx.nao()) throw std::runtime_error("libcint AO ordering mismatch");
    return shells;
}

// libcint stores the first shell index fastest. Initialize the output because
// screened shell quartets may leave it untouched.
std::vector<double> quartet(const LibcintWrapper& tables,
                            const std::array<FINT, 4>& ids,
                            const std::array<std::size_t, 4>& sizes)
{
    std::size_t count = 1;
    for (auto n : sizes) {
        if (n && count > std::numeric_limits<std::size_t>::max()/n)
            throw std::overflow_error("ERI shell quartet is too large");
        count *= n;
    }
    std::vector<double> values(count, 0.0);
    tables.int2e(values.data(), tables.dimensions(ids), ids);
    return values;
}
}

// Shared integral-direct build. Greedy and uniform differ only in pivot
// selection; both retain the global maximum for the stopping criterion.
ERICholesky cholesky_eri_impl(const LibcintContext& ctx, double tolerance,
                              std::size_t max_rank, bool debug,
                              CholeskyAlgorithm algorithm, std::uint64_t seed)
{
    if (!std::isfinite(tolerance) || tolerance < 0)
        throw std::invalid_argument("Cholesky tolerance must be finite and nonnegative");
    const auto nao = ctx.nao();
    if (nao > 0 && nao > (std::numeric_limits<std::size_t>::max()/nao - 1)/2)
        throw std::overflow_error("AO-pair dimension overflows size_t");
    const auto npair = nao*(nao+1)/2;
    if (npair == 0) throw std::invalid_argument("empty AO basis");
    if (max_rank > npair) throw std::invalid_argument("max_rank exceeds AO-pair dimension");
    const auto shells = shells_for(ctx);
    std::vector<std::pair<std::size_t, std::size_t>> shell_pairs;
    for (std::size_t s = 0; s < shells.size(); ++s)
        for (std::size_t t = 0; t <= s; ++t)
            shell_pairs.emplace_back(s, t);
    const auto& tables = ctx.tables();
    ERICholesky result;
    result.npair = npair;
    result.residual.assign(npair, 0.0);
    std::vector<std::size_t> ao_shell(nao);
    for (std::size_t s = 0; s < shells.size(); ++s)
        for (std::size_t a = shells[s].offset; a < shells[s].offset + shells[s].size; ++a)
            ao_shell[a] = s;

    const auto diagonal_started = std::chrono::steady_clock::now();
    parallel_shell_tasks("isdf_eri_diagonal", shell_pairs.size(), [&](std::size_t task) {
        const auto [s, t] = shell_pairs[task];
        const auto& a = shells[s];
        const auto& b = shells[t];
        const auto block = quartet(tables, {a.id,b.id,a.id,b.id},
                                   {a.size,b.size,a.size,b.size});
        for (std::size_t i = 0; i < a.size; ++i)
            for (std::size_t j = 0; j < b.size; ++j) {
                const auto mu = a.offset+i, nu = b.offset+j;
                if (mu < nu) continue;
                const auto value = block[i + a.size*(j + b.size*(i + a.size*j))];
                if (!std::isfinite(value) || value < -1e-12)
                    throw std::runtime_error("invalid ERI diagonal");
                result.residual[pair_index(mu,nu)] = std::max(0.0, value);
            }
    });
    const auto first = std::max_element(result.residual.begin(), result.residual.end());
    const double initial = *first;
    const double cutoff = tolerance * initial;
    const std::size_t cap = max_rank ? max_rank : npair;
    const auto* name = algorithm == CholeskyAlgorithm::UnblockedUniform ?
                       "unblocked_uniform" : "unblocked_greedy";
    if (debug) {
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - diagonal_started).count();
        std::cerr << "[cholesky ao] algorithm=" << name
                  << " shells=" << shells.size() << " pairs=" << npair
                  << " diagonal_quartets=" << shell_pairs.size()
                  << " initial_max_residual=" << initial << " cutoff=" << cutoff
                  << " diagonal_seconds=" << seconds << std::endl;
    }
    std::mt19937_64 generator(seed);
    CholeskyCandidate next_max{*first,
                                static_cast<std::size_t>(first-result.residual.begin()), false};
    while (result.pivots.size() < cap) {
        if (next_max.residual <= cutoff || next_max.residual <= 0.0) break;
        const auto selected = choose_cholesky_pivot(result.residual, next_max,
                                                     cutoff, algorithm, generator);
        const auto pivot = selected.index;
        const double pivot_residual = selected.residual;
        // Invert the packed index without floating-point rounding.
        std::size_t lo = 0, hi = nao;
        while (lo + 1 < hi) {
            const auto mid = lo + (hi-lo)/2;
            if (pair_index(mid,0) <= pivot) lo = mid; else hi = mid;
        }
        const auto mu = lo, nu = pivot - pair_index(mu,0);
        const auto& a = shells[ao_shell[mu]];
        const auto& b = shells[ao_shell[nu]];
        std::vector<double> column(npair, 0.0);
        const auto column_started = std::chrono::steady_clock::now();
        parallel_shell_tasks("isdf_eri_pivot_column", shell_pairs.size(),
                             [&](std::size_t task) {
            const auto [s, t] = shell_pairs[task];
            const auto& c = shells[s];
            const auto& d = shells[t];
            const auto block = quartet(tables, {c.id,d.id,a.id,b.id},
                                       {c.size,d.size,a.size,b.size});
            const auto ai = mu-a.offset, bi = nu-b.offset;
            for (std::size_t i = 0; i < c.size; ++i)
                for (std::size_t j = 0; j < d.size; ++j) {
                    const auto x = c.offset+i, y = d.offset+j;
                    if (x < y) continue;
                    column[pair_index(x,y)] =
                        block[i + c.size*(j + d.size*(ai + a.size*bi))];
                }
        });
        const auto column_finished = std::chrono::steady_clock::now();
        next_max = correct_cholesky_column(column, result.factors, result.residual,
                                           pivot, pivot_residual,
                                           "Cholesky factor dimensions overflow size_t",
                                           "non-finite Cholesky vector");
        result.pivots.push_back(pivot);
        if (debug) {
            const auto integral_seconds = std::chrono::duration<double>(
                column_finished - column_started).count();
            const auto update_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - column_finished).count();
            std::cerr << "[cholesky ao] rank=" << result.pivots.size()
                      << " pivot=(" << mu << ',' << nu << ") shell_pair=("
                      << a.id << ',' << b.id << ") pivot_residual=" << pivot_residual
                      << " max_remaining=" << next_max.residual
                      << " column_quartets=" << shell_pairs.size()
                      << " integral_seconds=" << integral_seconds
                      << " update_seconds=" << update_seconds << std::endl;
        }
    }
    if (debug) {
        const auto remaining = next_max.residual;
        std::cerr << "[cholesky ao] stop rank=" << result.pivots.size()
                  << " max_remaining=" << remaining
                  << " reason=" << (remaining == 0.0 ? "zero_residual" :
                      remaining <= cutoff ? "tolerance" : "rank_limit") << std::endl;
    }
    return result;
}

ERICholesky cholesky_eri(const LibcintContext& ctx, double tolerance,
                         std::size_t max_rank, bool debug)
{
    return cholesky_eri_impl(ctx, tolerance, max_rank, debug,
                             CholeskyAlgorithm::UnblockedGreedy, 0);
}
} // namespace isdf
