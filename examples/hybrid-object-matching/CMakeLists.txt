# Example 5 -- hybrid object matching.
#
# The advanced workload: MPI across ranks, OpenMP or CUDA inside each one, and
# a determinism contract that lets all four be asserted byte-identical. It was
# the whole project before the repository became an MPI environment, which is
# why it carries its own include/, src/, tests/, tools/ and bench/ rather than
# being a single main.c like the other examples.
#
# Toolchain detection happens once at the repository root; this file only reads
# MPILAB_HAVE_* and links what was found.

set(MSEARCH_VERSION_STR "1.0.0")

set(MSEARCH_SOURCES
  src/core/log.c
  src/core/problem.c
  src/core/parser.c
  src/core/writer.c
  src/backends/registry.c
  src/backends/backend_serial.c
  src/runtime/runner_local.c
  src/cli/options.c)

set(MSEARCH_DEFINES "")
set(MSEARCH_LIBS "")

if(MPILAB_HAVE_OPENMP)
  list(APPEND MSEARCH_SOURCES src/backends/backend_openmp.c)
  list(APPEND MSEARCH_DEFINES MSEARCH_HAVE_OPENMP)
  list(APPEND MSEARCH_LIBS OpenMP::OpenMP_C)
endif()

if(MPILAB_HAVE_MPI)
  list(APPEND MSEARCH_SOURCES src/runtime/runner_mpi.c)
  list(APPEND MSEARCH_DEFINES MSEARCH_HAVE_MPI)
  list(APPEND MSEARCH_LIBS MPI::MPI_C)
endif()

if(MPILAB_HAVE_CUDA)
  list(APPEND MSEARCH_SOURCES src/backends/backend_cuda.cu)
  list(APPEND MSEARCH_DEFINES MSEARCH_HAVE_CUDA)
endif()

# ------------------------------------------------------------------- targets
add_library(msearch_core STATIC ${MSEARCH_SOURCES})
target_include_directories(msearch_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(msearch_core PUBLIC ${MSEARCH_DEFINES}
  PRIVATE MSEARCH_VERSION="${MSEARCH_VERSION_STR}")
target_link_libraries(msearch_core PUBLIC ${MSEARCH_LIBS} PRIVATE mpilab_warnings)
if(NOT MSVC)
  target_link_libraries(msearch_core PUBLIC m)
endif()

add_executable(msearch src/cli/main.c)
target_link_libraries(msearch PRIVATE msearch_core mpilab_warnings)

# This example is the one where the directory and the binary have different
# names, so scripts/run-mpi.sh is told about both -- `run-mpi.sh -n 4
# examples/hybrid-object-matching` and `run-mpi.sh -n 4 msearch` then mean the
# same thing.
mpilab_register_example(hybrid-object-matching msearch
  "MPI + OpenMP + CUDA submatrix search, verified byte-identical across backends")

if(MPILAB_BUILD_TESTS)
  add_subdirectory(tests)
endif()
