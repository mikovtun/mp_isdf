#include "anchor_select.hpp"

#include <Kokkos_Core.hpp>
#include <lapack.hh>

#ifdef ISDF_ENABLE_CUDA
#include <cuda_runtime.h>
#include <magma_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace isdf {
namespace {

void check_product_size(std::size_t nfeatures, std::size_t npoints)
{
    if (nfeatures == 0 || nfeatures > std::numeric_limits<std::size_t>::max() / nfeatures ||
        npoints > std::numeric_limits<std::size_t>::max() / (nfeatures * nfeatures) ||
        nfeatures * nfeatures > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("pair-product matrix is too large");
    }
}

#ifndef ISDF_ENABLE_CUDA
std::vector<std::int64_t> qrcp_cpu(std::vector<double>& matrix,
                                   std::size_t m, std::size_t n)
{
    std::vector<std::int64_t> jpvt(n, 0);
    std::vector<double> tau(std::min(m, n));
    if (lapack::geqp3(static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
                      matrix.data(), static_cast<std::int64_t>(m), jpvt.data(), tau.data()) != 0) {
        throw std::runtime_error("LAPACK++ geqp3 failed");
    }
    return jpvt; // LAPACK uses 1-based pivot indices.
}
#endif

#ifdef ISDF_ENABLE_CUDA
std::vector<std::int64_t> qrcp_magma(double* device_matrix,
                                     std::size_t m, std::size_t n)
{
    if (m > static_cast<std::size_t>(std::numeric_limits<magma_int_t>::max()) ||
        n > static_cast<std::size_t>(std::numeric_limits<magma_int_t>::max())) {
        throw std::overflow_error("MAGMA matrix dimensions exceed magma_int_t");
    }
    if (magma_init() != MAGMA_SUCCESS) throw std::runtime_error("magma_init failed");
    struct Finalize { ~Finalize() { magma_finalize(); } } finalize;
    const auto rows = static_cast<magma_int_t>(m);
    const auto cols = static_cast<magma_int_t>(n);
    std::vector<magma_int_t> pivots(n, 0);
    std::vector<double> tau(std::min(m, n));
    magma_device_t device;
    magma_getdevice(&device);
    magma_queue_t queue;
    magma_queue_create(device, &queue);
    struct DestroyQueue {
        magma_queue_t queue;
        ~DestroyQueue() { magma_queue_destroy(queue); }
    } destroy_queue{queue};

    magma_int_t host_bytes = -1, device_bytes = -1, info = 0;
    magma_dgeqp3_expert_gpu_work(rows, cols, nullptr, rows, nullptr, nullptr,
                                  nullptr, &host_bytes, nullptr, &device_bytes, &info, queue);
    if (info != 0 || host_bytes < 0 || device_bytes <= 0) {
        throw std::runtime_error("MAGMA geqp3 GPU workspace query failed");
    }
    void* workspace = nullptr;
    if (magma_malloc(&workspace, static_cast<std::size_t>(device_bytes)) != MAGMA_SUCCESS) {
        throw std::runtime_error("MAGMA geqp3 GPU workspace allocation failed");
    }
    struct FreeWorkspace {
        void* ptr;
        ~FreeWorkspace() { magma_free(ptr); }
    } free_workspace{workspace};
    // The expert API's lengths are bytes; unlike magma_dgeqp3_gpu its host
    // length must be reset to zero after querying, or it queries again.
    magma_dgeqp3_expert_gpu_work(rows, cols, device_matrix, rows, pivots.data(), tau.data(),
                                  nullptr, &host_bytes, workspace, &device_bytes, &info, queue);
    magma_queue_sync(queue);
    if (info != 0) throw std::runtime_error("MAGMA geqp3 GPU failed: " + std::to_string(info));
    return {pivots.begin(), pivots.end()}; // MAGMA also uses 1-based indices.
}
#endif

} // namespace

std::vector<std::size_t> AnchorSelection::select(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights,
    const std::vector<AxisType>& axes,
    double tolerance,
    std::size_t max_rank,
    LocalAnchorSelectionAlgorithm algorithm,
    const std::vector<double>& mo_coeff,
    std::size_t nocc) const
{
    if (algorithm != LocalAnchorSelectionAlgorithm::QRColumnPivoting) {
        throw std::logic_error("randomized QRCP is not implemented");
    }
    if (points.size() != weights.size() || axes.empty() ||
        !std::isfinite(tolerance) || tolerance < 0) {
        throw std::invalid_argument("invalid weights, axis list, or tolerance");
    }
    const std::size_t nao = context_.nao();
    bool need_ao = false, need_grad = false, need_occ = false;
    std::size_t nfeatures = 0;
    for (AxisType axis : axes) {
        std::size_t width = 0;
        switch (axis) {
        case AxisType::AtomicOrbital: need_ao = true; width = nao; break;
        case AxisType::AtomicOrbitalGradX:
        case AxisType::AtomicOrbitalGradY:
        case AxisType::AtomicOrbitalGradZ: need_grad = true; width = nao; break;
        case AxisType::OccupiedOrbital: need_occ = true; width = nocc; break;
        default: throw std::invalid_argument("unknown AxisType");
        }
        if (width > std::numeric_limits<std::size_t>::max() - nfeatures) {
            throw std::overflow_error("axis feature count is too large");
        }
        nfeatures += width;
    }
    if (need_occ) {
        if (nao == 0 || nocc == 0 || nocc > nao ||
            nao > std::numeric_limits<std::size_t>::max() / nao ||
            mo_coeff.size() != nao * nao) {
            throw std::invalid_argument("OccupiedOrbital requires row-major (nao,nao) RHF mo_coeff and 1 <= nocc <= nao");
        }
        need_ao = true;
    }
    if (nfeatures == 0) throw std::invalid_argument("no orbital features available");

    std::vector<std::size_t> original;
    std::vector<std::array<double, 3>> kept_points;
    std::vector<double> kept_weights;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!std::isfinite(weights[i])) throw std::invalid_argument("grid weights must be finite");
        if (weights[i] == 0) continue;
        for (double x : points[i]) {
            if (!std::isfinite(x)) throw std::invalid_argument("grid coordinates must be finite");
        }
        original.push_back(i);
        kept_points.push_back(points[i]);
        kept_weights.push_back(weights[i]);
    }
    if (kept_points.empty()) return {};
    check_product_size(nfeatures, kept_points.size());
    const auto ao = need_ao ? context_.eval_ao(kept_points) : std::vector<double>{};
    const auto grad = need_grad ? context_.eval_ao_gradient(kept_points) : std::vector<double>{};
    const std::size_t ng = kept_points.size();
    std::vector<double> features(ng * nfeatures);
    for (std::size_t g = 0; g < ng; ++g) {
        std::size_t column = 0;
        for (AxisType axis : axes) {
            switch (axis) {
            case AxisType::AtomicOrbital:
                std::copy_n(ao.data() + g * nao, nao, features.data() + g*nfeatures + column);
                column += nao;
                break;
            case AxisType::AtomicOrbitalGradX:
            case AxisType::AtomicOrbitalGradY:
            case AxisType::AtomicOrbitalGradZ: {
                const std::size_t component = static_cast<std::size_t>(axis) -
                                              static_cast<std::size_t>(AxisType::AtomicOrbitalGradX);
                std::copy_n(grad.data() + (component * ng + g) * nao, nao,
                            features.data() + g*nfeatures + column);
                column += nao;
                break;
            }
            case AxisType::OccupiedOrbital:
                for (std::size_t j = 0; j < nocc; ++j) {
                    double value = 0.0;
                    for (std::size_t k = 0; k < nao; ++k) {
                        value += ao[g * nao + k] * mo_coeff[k * nao + j];
                    }
                    features[g*nfeatures + column + j] = value;
                }
                column += nocc;
                break;
            }
        }
    }

    // A = weighted product^T; columns of A correspond to grid points.
    // Kokkos writes columns contiguously so LAPACK/MAGMA can consume them.
    const std::size_t m = nfeatures * nfeatures;
#ifdef ISDF_ENABLE_CUDA
    // The MAGMA input must be in CUDA memory even when another Kokkos backend
    // happens to be the default execution space.
    using Device = Kokkos::Cuda;
#else
    using Device = Kokkos::DefaultExecutionSpace;
#endif
    Kokkos::View<double*, Device> d_features("axis_features", features.size());
    Kokkos::View<double*, Device> d_weights("grid_weights", ng);
    Kokkos::View<double*, Device> d_product("pair_product_transpose", ng * m);
    auto h_features = Kokkos::create_mirror_view(d_features);
    auto h_weights = Kokkos::create_mirror_view(d_weights);
    for (std::size_t i = 0; i < features.size(); ++i) h_features(i) = features[i];
    for (std::size_t i = 0; i < ng; ++i) h_weights(i) = std::sqrt(std::abs(kept_weights[i]));
    Kokkos::deep_copy(d_features, h_features);
    Kokkos::deep_copy(d_weights, h_weights);
    Kokkos::parallel_for("isdf_pair_product", Kokkos::RangePolicy<Device>(0, ng*m),
                         KOKKOS_LAMBDA(const std::size_t index) {
        const auto g = index / m;
        const auto pair = index % m;
        d_product(index) = d_weights(g) * d_features(g*nfeatures + pair/nfeatures) *
                           d_features(g*nfeatures + pair%nfeatures);
    });
    int nonfinite = 0;
    Kokkos::parallel_reduce("isdf_check_pair_product", Kokkos::RangePolicy<Device>(0, ng*m),
                            KOKKOS_LAMBDA(const std::size_t i, int& bad) {
        if (!Kokkos::isfinite(d_product(i))) ++bad;
    }, nonfinite);
    if (nonfinite) throw std::overflow_error("non-finite weighted pair product");

    const auto cap = std::min({m, ng, max_rank ? max_rank : std::min(m, ng)});
    std::vector<double> diagonal(cap);
    std::vector<std::int64_t> jpvt;
#ifdef ISDF_ENABLE_CUDA
    // MAGMA uses its own stream; finish the Kokkos product kernel before QR.
    Kokkos::fence();
    jpvt = qrcp_magma(d_product.data(), m, ng);
    Kokkos::View<double*, Device> d_diagonal("qr_diagonal", cap);
    Kokkos::parallel_for("isdf_qr_diagonal", Kokkos::RangePolicy<Device>(0, cap),
                         KOKKOS_LAMBDA(const std::size_t i) {
        d_diagonal(i) = d_product(i*m + i);
    });
    auto h_diagonal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_diagonal);
    for (std::size_t i = 0; i < cap; ++i) diagonal[i] = h_diagonal(i);
#else
    auto h_product = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_product);
    std::vector<double> matrix(ng * m);
    for (std::size_t i = 0; i < matrix.size(); ++i) matrix[i] = h_product(i);
    jpvt = qrcp_cpu(matrix, m, ng);
    for (std::size_t i = 0; i < cap; ++i) diagonal[i] = matrix[i*m + i];
#endif
    const double first = cap ? std::abs(diagonal[0]) : 0.0;
    std::vector<std::size_t> selected;
    for (std::size_t i = 0; i < cap; ++i) {
        if (first == 0 || std::abs(diagonal[i]) <= tolerance * first) break;
        const auto pivot = jpvt[i];
        if (pivot < 1 || static_cast<std::size_t>(pivot) > ng) {
            throw std::runtime_error("QRCP returned an invalid grid pivot");
        }
        selected.push_back(original[static_cast<std::size_t>(pivot - 1)]);
    }
    return selected;
}

std::vector<std::size_t> DistributedAnchorSelection::select(
    const std::vector<std::array<double, 3>>&,
    const std::vector<double>&,
    const std::vector<AxisType>&,
    DistributedAnchorSelectionAlgorithm) const
{
    throw std::logic_error("distributed SLATE SVD anchor selection is not implemented");
}

} // namespace isdf
