#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <exception>
#include <mutex>

namespace isdf {

// libcint shell quartets are evaluated on the host. Each task must own its
// integral output/cache and write a disjoint range of the destination array.
// Do not let a C++ exception escape an OpenMP/Kokkos worker thread.
template <class Function>
void parallel_shell_tasks(const char* label, std::size_t count, Function&& task)
{
    if (count < 8) {
        for (std::size_t i = 0; i < count; ++i) task(i);
        return;
    }
    std::exception_ptr failure;
    std::mutex failure_mutex;
    Kokkos::DefaultHostExecutionSpace exec;
    Kokkos::parallel_for(label,
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(exec, 0, count),
        [&](std::size_t i) {
            try {
                task(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
            }
        });
    exec.fence();
    if (failure) std::rethrow_exception(failure);
}

} // namespace isdf
