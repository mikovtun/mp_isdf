cmake_minimum_required(VERSION 3.22)
project(mp_isdf LANGUAGES CXX)

option(ISDF_ENABLE_MPI "Enable MPI support" ON)
option(ISDF_ENABLE_CUDA "Enable CUDA support" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Kokkos and libcint are also linked into a Python extension module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# The large randomized sketch path needs 64-bit BLAS/LAPACK integer ABIs.
# BLAS++ uses this cache option for its C++ wrappers; FindLAPACK uses BLA_*
# when MAGMA and LAPACK++ locate the Fortran backend.
set(blas_int "int64_t (ILP64)" CACHE STRING "BLAS integer size" FORCE)
set(use_cmake_find_lapack ON CACHE BOOL "Use CMake FindLAPACK for ILP64 selection" FORCE)
set(BLA_SIZEOF_INTEGER 8 CACHE STRING "Require ILP64 BLAS/LAPACK integers" FORCE)
# Select a known ILP64 MKL interface where available instead of allowing
# BLAS++'s generic finder to pair an integer-width-ambiguous BLAS with ILP64 LAPACK.
set(BLA_VENDOR Intel10_64ilp_seq CACHE STRING "Select ILP64 MKL BLAS/LAPACK" FORCE)
FetchContent_MakeAvailable(blaspp lapackpp)
# The embedded LAPACK++ configuration can omit the Fortran symbol-mangling
# definition when a vendor BLAS/LAPACK is found via CMake. ILP64 libraries
# retain the conventional trailing-underscore symbols.
target_compile_definitions(lapackpp PUBLIC LAPACK_FORTRAN_ADD_ LAPACK_ILP64)
target_compile_definitions(blaspp PUBLIC BLAS_FORTRAN_ADD_)
# Some BLAS++ CMake finder paths generate BLAS_ILP64 in defines.h, while
# cached CMake-FindBLAS configurations may omit it. Ensure ABI declarations
# are 64-bit in either case without redefining a generated header macro.
if(NOT "-DBLAS_ILP64" IN_LIST blaspp_defines)
    target_compile_definitions(blaspp PUBLIC BLAS_ILP64)
endif()

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

find_package(LAPACK REQUIRED)
if(NOT BLA_SIZEOF_INTEGER STREQUAL "8")
    message(WARNING "LAPACK was not configured with the requested ILP64 integer ABI; "
                    "large-matrix indices may be limited to 32-bit offsets")
endif()
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
    check_cxx_source_runs("extern \"C\" void dgesv_(long long*, long long*, double*, long long*, long long*, double*, long long*, long long*);\nint main() { long long n=2,nrhs=1,lda=2,ldb=2,ipiv[2]={0,0},info=-999; double a[4]={2,1,1,3},b[2]={1,2}; dgesv_(&n,&nrhs,a,&lda,ipiv,b,&ldb,&info); return !(info==0 && b[0]>0.19 && b[0]<0.21 && b[1]>0.59 && b[1]<0.61); }"
    ISDF_LAPACK_ILP64_ABI_WORKS)
unset(CMAKE_REQUIRED_LIBRARIES)
if(NOT ISDF_LAPACK_ILP64_ABI_WORKS)
    message(WARNING "Could not verify the linked LAPACK ILP64 ABI at configure time. "
                    "Ensure that the selected library really uses 64-bit Fortran integers; "
                    "otherwise large QRCP matrices are unsafe.")
else()
    message(STATUS "Verified LAPACK ILP64 integer ABI")
endif()

if(ISDF_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    # MAGMA uses CMAKE_SOURCE_DIR internally, so it must be configured as a
    # standalone project rather than added with FetchContent_MakeAvailable.
    include(ExternalProject)
    # MAGMA's Git checkout omits generated precision sources and CMake.src.cuda.
    # Its Makefile generates both before the standalone CMake configure step.
    find_program(ISDF_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
    set(ISDF_MAGMA_GENERATE_ARCH sm_80)
    if(CMAKE_CUDA_ARCHITECTURES)
        list(GET CMAKE_CUDA_ARCHITECTURES 0 ISDF_MAGMA_FIRST_ARCH)
        if(ISDF_MAGMA_FIRST_ARCH MATCHES "^([0-9]+)")
            set(ISDF_MAGMA_GENERATE_ARCH "sm_${CMAKE_MATCH_1}")
        endif()
    endif()
    # Pass the exact ILP64 LAPACK chosen above through to MAGMA's standalone
    # configure; otherwise its independent FindLAPACK can silently select LP64.
    string(REPLACE ";" "|" ISDF_MAGMA_LAPACK_LIBRARIES "${LAPACK_LIBRARIES}")
    set(ISDF_MAGMA_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/magma-src")
    set(ISDF_MAGMA_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/magma-build")
    set(ISDF_MAGMA_LIBRARY "${ISDF_MAGMA_BINARY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}magma${CMAKE_STATIC_LIBRARY_SUFFIX}")
    file(MAKE_DIRECTORY "${ISDF_MAGMA_SOURCE_DIR}/include" "${ISDF_MAGMA_BINARY_DIR}/include")
    ExternalProject_Add(magma_external
        LIST_SEPARATOR "|"
        GIT_REPOSITORY https://github.com/icl-utk-edu/magma.git
        GIT_TAG v2.10.0
        GIT_SHALLOW TRUE
        SOURCE_DIR "${ISDF_MAGMA_SOURCE_DIR}"
        BINARY_DIR "${ISDF_MAGMA_BINARY_DIR}"
        PATCH_COMMAND ${ISDF_MAKE_EXECUTABLE} -C <SOURCE_DIR> generate
            FORT=true BACKEND=cuda GPU_TARGET=${ISDF_MAGMA_GENERATE_ARCH}
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DMAGMA_ENABLE_CUDA=ON
            -DMAGMA_ENABLE_HIP=OFF
            -DUSE_FORTRAN=OFF
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_C_FLAGS=-DMAGMA_ILP64
            -DCMAKE_CXX_FLAGS=-DMAGMA_ILP64
            -DCMAKE_CUDA_FLAGS=-DMAGMA_ILP64
            -DLAPACK_LIBRARIES:STRING=${ISDF_MAGMA_LAPACK_LIBRARIES}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target magma
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${ISDF_MAGMA_LIBRARY}"
    )
    add_library(isdf_magma STATIC IMPORTED GLOBAL)
    set_target_properties(isdf_magma PROPERTIES
        IMPORTED_LOCATION "${ISDF_MAGMA_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ISDF_MAGMA_SOURCE_DIR}/include;${ISDF_MAGMA_BINARY_DIR}/include"
        INTERFACE_COMPILE_DEFINITIONS MAGMA_ILP64
    )
    add_dependencies(isdf_magma magma_external)
    target_link_libraries(isdf_magma INTERFACE
        ${LAPACK_LIBRARIES} CUDA::cudart CUDA::cublas CUDA::cusparse)
    if(ISDF_ENABLE_OPENMP)
        target_link_libraries(isdf_magma INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

add_executable(mp_isdf cpp/src/main.cpp)
target_include_directories(mp_isdf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

# SLATE package exports the `slate` target in current releases; accept the
# namespaced target as well for packages that provide it.
if(TARGET SLATE::slate)
    set(SLATE_LIBRARY_TARGET SLATE::slate)
elseif(TARGET slate)
    set(SLATE_LIBRARY_TARGET slate)
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

add_library(mp_isdf_orbitals STATIC cpp/src/ao_eval.cpp)
target_include_directories(mp_isdf_orbitals PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_link_libraries(mp_isdf_orbitals PUBLIC Kokkos::kokkos ${ISDF_CINT_TARGET})
set_target_properties(mp_isdf_orbitals PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(isdf_anchor_core STATIC
    cpp/src/anchor_select.cpp
    cpp/src/anchor_select_qrcp.cpp
    cpp/src/anchor_select_online_qr.cpp
    cpp/src/anchor_select_randomized_id.cpp
    cpp/src/anchor_select_cvt.cpp)
target_include_directories(isdf_anchor_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_link_libraries(isdf_anchor_core PUBLIC mp_isdf_orbitals lapackpp)
set_target_properties(isdf_anchor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ISDF_ENABLE_CUDA)
    target_link_libraries(isdf_anchor_core PUBLIC isdf_magma CUDA::cudart)
    target_compile_definitions(isdf_anchor_core PRIVATE ISDF_ENABLE_CUDA=1)
endif()

add_library(isdf_coulomb_core STATIC
    cpp/src/cholesky_eri_greedy.cpp
    cpp/src/cholesky_eri_uniform.cpp
    cpp/src/coulomb_metric.cpp
    cpp/src/derivative_metric.cpp)
target_include_directories(isdf_coulomb_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_link_libraries(isdf_coulomb_core PUBLIC mp_isdf_orbitals blaspp)
set_target_properties(isdf_coulomb_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# pybind11's default LTO can duplicate CUDA fatbinData during device linking.
# The extension and executable both use the public name mp_isdf, but CMake
# targets must be distinct; OUTPUT_NAME sets the Python import name.
pybind11_add_module(mp_isdf_python NO_EXTRAS cpp/src/ao_eval_pybind.cpp cpp/src/anchor_select_pybind.cpp cpp/src/coulomb_metric_pybind.cpp)
set_target_properties(mp_isdf_python PROPERTIES OUTPUT_NAME mp_isdf)
target_link_libraries(mp_isdf_python PRIVATE isdf_anchor_core isdf_coulomb_core)

target_link_libraries(mp_isdf PRIVATE
    Kokkos::kokkos
    ${SLATE_LIBRARY_TARGET}
    ${ISDF_CINT_TARGET}
    pybind11::embed
)

if(ISDF_ENABLE_MPI)
    target_link_libraries(mp_isdf PRIVATE MPI::MPI_CXX)
    target_compile_definitions(mp_isdf PRIVATE ISDF_ENABLE_MPI=1)
endif()

if(ISDF_ENABLE_CUDA)
    target_link_libraries(mp_isdf PRIVATE CUDA::cudart isdf_magma)
    target_compile_definitions(mp_isdf PRIVATE ISDF_ENABLE_CUDA=1)
endif()
