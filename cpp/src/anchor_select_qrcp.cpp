#include "anchor_select_qrcp.hpp"
#include "anchor_select.hpp"
#include "anchor_select_qrcp_backend.hpp"

#include <Kokkos_Core.hpp>
#include <lapack.hh>

#ifdef ISDF_ENABLE_CUDA
#include <magma_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
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

} // namespace
namespace detail {

std::vector<std::int64_t> qrcp_cpu(double* matrix, std::size_t m, std::size_t n)
{
    std::vector<std::int64_t> jpvt(n, 0);
    std::vector<double> tau(std::min(m, n));
    if (lapack::geqp3(static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
                      matrix, static_cast<std::int64_t>(m), jpvt.data(), tau.data()) != 0) {
        throw std::runtime_error("LAPACK++ geqp3 failed");
    }
    return jpvt; // LAPACK uses 1-based pivot indices.
}

#ifdef ISDF_ENABLE_CUDA
std::vector<std::int64_t> qrcp_magma(double* device_matrix, std::size_t m, std::size_t n)
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

} // namespace detail

std::vector<std::size_t> QRCPAnchorSelection::select(
    const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& points,
    const std::vector<double>& weights,
    const std::vector<AxisType>& axes,
    double tolerance, std::size_t max_rank,
    const std::vector<double>& mo_coeff, std::size_t nocc,
    EvaluationSpace space)
{
    if (points.size() != weights.size() ||
        !std::isfinite(tolerance) || tolerance < 0) {
        throw std::invalid_argument("invalid weights, axis list, or tolerance");
    }
    switch (space) {
    case EvaluationSpace::Auto:
    case EvaluationSpace::Host: break;
    case EvaluationSpace::Device:
#ifndef ISDF_ENABLE_CUDA
        throw std::invalid_argument("device evaluation requires a CUDA build");
#else
        break;
#endif
    default: throw std::invalid_argument("unknown evaluation space");
    }
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
    const auto spec = selector.axis_spec(axes, mo_coeff, nocc);
    if (kept_points.empty()) return {};
    check_product_size(spec.width, kept_points.size());
    switch (space) {
    case EvaluationSpace::Host:
        return select_in_space<Kokkos::DefaultHostExecutionSpace>(
            selector, kept_points, kept_weights, original, axes, tolerance, max_rank, mo_coeff, nocc);
    case EvaluationSpace::Auto:
#ifdef ISDF_ENABLE_CUDA
    case EvaluationSpace::Device:
        return select_in_space<Kokkos::Cuda>(
            selector, kept_points, kept_weights, original, axes, tolerance, max_rank, mo_coeff, nocc);
#else
        return select_in_space<Kokkos::DefaultHostExecutionSpace>(
            selector, kept_points, kept_weights, original, axes, tolerance, max_rank, mo_coeff, nocc);
    case EvaluationSpace::Device:
        throw std::invalid_argument("device evaluation requires a CUDA build");
#endif
    }
    throw std::invalid_argument("unknown evaluation space");
}

template <class Device>
std::vector<std::size_t> QRCPAnchorSelection::select_in_space(
    const AnchorSelection& selector,
    const std::vector<std::array<double, 3>>& kept_points,
    const std::vector<double>& kept_weights,
    const std::vector<std::size_t>& original,
    const std::vector<AxisType>& axes,
    double tolerance, std::size_t max_rank,
    const std::vector<double>& mo_coeff, std::size_t nocc)
{
    const auto axis = selector.evaluate_axis<Device>(kept_points, axes, mo_coeff, nocc);
    const std::size_t nfeatures = axis.width;
    const auto& features = axis.values;
    const std::size_t ng = kept_points.size();

    // A = weighted product^T; columns of A correspond to grid points.
    // Kokkos writes columns contiguously so LAPACK/MAGMA can consume them.
    const std::size_t m = nfeatures * nfeatures;
    Kokkos::View<double*, Device> d_weights("grid_weights", ng);
    Kokkos::View<double*, Device> d_product("pair_product_transpose", ng * m);
    auto h_weights = Kokkos::create_mirror_view(d_weights);
    for (std::size_t i = 0; i < ng; ++i) h_weights(i) = std::sqrt(std::abs(kept_weights[i]));
    Kokkos::deep_copy(d_weights, h_weights);
    Kokkos::parallel_for("isdf_pair_product", Kokkos::RangePolicy<Device>(0, ng*m),
                         KOKKOS_LAMBDA(const std::size_t index) {
        const auto g = index / m;
        const auto pair = index % m;
        d_product(index) = d_weights(g) * features(g*nfeatures + pair/nfeatures) *
                           features(g*nfeatures + pair%nfeatures);
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
    // MAGMA consumes the device matrix in place; LAPACK++ consumes the host
    // view in place. Neither path copies the entire product matrix across PCIe.
    Kokkos::fence();
#ifdef ISDF_ENABLE_CUDA
    if constexpr (std::is_same_v<Device, Kokkos::Cuda>) {
        jpvt = detail::qrcp_magma(d_product.data(), m, ng);
        Kokkos::View<double*, Device> d_diagonal("qr_diagonal", cap);
        Kokkos::parallel_for("isdf_qr_diagonal", Kokkos::RangePolicy<Device>(0, cap),
                             KOKKOS_LAMBDA(const std::size_t i) {
            d_diagonal(i) = d_product(i*m + i);
        });
        auto h_diagonal = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d_diagonal);
        for (std::size_t i = 0; i < cap; ++i) diagonal[i] = h_diagonal(i);
    } else
#endif
    {
        jpvt = detail::qrcp_cpu(d_product.data(), m, ng);
        for (std::size_t i = 0; i < cap; ++i) diagonal[i] = d_product(i*m + i);
    }
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

} // namespace isdf
