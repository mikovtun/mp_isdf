#pragma once

#include <Kokkos_Core.hpp>
#include <slate/slate.hh>
#include <pybind11/embed.h>

#ifdef ISDF_ENABLE_MPI
#include <mpi.h>
#endif

#ifdef ISDF_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
