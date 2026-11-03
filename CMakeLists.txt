cmake_minimum_required(VERSION 3.21)
project(isdf_slate LANGUAGES CXX)

option(ISDF_ENABLE_MPI "Enable MPI support" ON)
option(ISDF_ENABLE_CUDA "Enable CUDA support" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Configure Kokkos before adding it so the requested backends are enabled.
option(ISDF_ENABLE_OPENMP "Enable Kokkos OpenMP backend" ON)
set(Kokkos_ENABLE_CUDA ${ISDF_ENABLE_CUDA} CACHE BOOL "Enable Kokkos CUDA backend" FORCE)
set(Kokkos_ENABLE_OPENMP ${ISDF_ENABLE_OPENMP} CACHE BOOL "Enable Kokkos OpenMP backend" FORCE)
set(Kokkos_ENABLE_MPI ${ISDF_ENABLE_MPI} CACHE BOOL "Enable Kokkos MPI support" FORCE)
FetchContent_Declare(
    kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 5.2.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(kokkos)

# Build SLATE's dependencies first so their CMake targets exist before SLATE
# checks for them (SLATE's fallback path checks CMAKE_SOURCE_DIR, not its own source dir).
FetchContent_Declare(
    blaspp
    GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp.git
    GIT_TAG v2025.05.28
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    lapackpp
    GIT_REPOSITORY https://github.com/icl-utk-edu/lapackpp.git
    GIT_TAG v2025.05.28
    GIT_SHALLOW TRUE
)
if(ISDF_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

FetchContent_MakeAvailable(blaspp lapackpp)

# SLATE enables its test suite by default when embedded as a dependency;
# those tests require source-tree-only headers and are not needed here.
set(build_tests OFF CACHE BOOL "Build SLATE tests" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable dependency test suites" FORCE)
FetchContent_Declare(
    slate
    GIT_REPOSITORY https://github.com/icl-utk-edu/slate.git
    GIT_TAG v2025.05.28
    # SLATE requires these source submodules; FetchContent does not initialize
    # submodules unless explicitly requested.
    GIT_SUBMODULES blaspp lapackpp
    GIT_SUBMODULES_RECURSE TRUE
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(slate)

FetchContent_Declare(
    libcint
    GIT_REPOSITORY https://github.com/sunqm/libcint.git
    GIT_TAG v6.2.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libcint)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)

if(ISDF_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(ISDF_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_executable(isdf_slate src/main.cpp)
target_include_directories(isdf_slate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# SLATE package exports the `slate` target in current releases; accept the
# namespaced target as well for packages that provide it.
if(TARGET SLATE::slate)
    set(ISDF_SLATE_TARGET SLATE::slate)
elseif(TARGET slate)
    set(ISDF_SLATE_TARGET slate)
else()
    message(FATAL_ERROR "SLATE was found, but it did not export SLATE::slate or slate")
endif()

if(TARGET cint)
    set(ISDF_CINT_TARGET cint)
elseif(TARGET libcint)
    set(ISDF_CINT_TARGET libcint)
else()
    message(FATAL_ERROR "libcint was fetched, but no cint or libcint target was exported")
endif()

target_link_libraries(isdf_slate PRIVATE
    Kokkos::kokkos
    ${ISDF_SLATE_TARGET}
    ${ISDF_CINT_TARGET}
    pybind11::embed
)

if(ISDF_ENABLE_MPI)
    target_link_libraries(isdf_slate PRIVATE MPI::MPI_CXX)
    target_compile_definitions(isdf_slate PRIVATE ISDF_ENABLE_MPI=1)
endif()

if(ISDF_ENABLE_CUDA)
    target_link_libraries(isdf_slate PRIVATE CUDA::cudart)
    target_compile_definitions(isdf_slate PRIVATE ISDF_ENABLE_CUDA=1)
endif()
