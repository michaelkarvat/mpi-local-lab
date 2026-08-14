cmake_minimum_required(VERSION 3.18)
project(mpi-local-lab
        VERSION 1.0.0
        DESCRIPTION "Local MPI development and simulation environment"
        LANGUAGES C)

# --------------------------------------------------------------------------
# This is the environment, not an application. The root does exactly two jobs:
# detect the optional toolchains once, and add the examples that use them.
#
# Detection rather than requirement is deliberate and predates this file: the
# serial path has no dependencies, so `cmake . && cmake --build .` produces
# working binaries on a bare laptop, and the same tree gains MPI, OpenMP and
# CUDA automatically inside the containers or on a real cluster. Every optional
# feature can also be forced off, which is what lets CI test each configuration
# in isolation.
# --------------------------------------------------------------------------
option(MPILAB_ENABLE_MPI    "Build MPI examples if MPI is found"          ON)
option(MPILAB_ENABLE_OPENMP "Use OpenMP where an example supports it"     ON)
option(MPILAB_ENABLE_CUDA   "Use CUDA where an example supports it"       ON)
option(MPILAB_BUILD_TESTS   "Build the test suite"                        ON)
option(MPILAB_WERROR        "Treat warnings as errors"                    OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Every runnable target lands in one directory. scripts/run-mpi.sh names an
# example and expects to find its binary without knowing how the source tree is
# organised, and mpirun needs the same path to resolve on every node.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_library(mpilab_warnings INTERFACE)
if(MSVC)
  target_compile_options(mpilab_warnings INTERFACE /W4)
  if(MPILAB_WERROR)
    target_compile_options(mpilab_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(mpilab_warnings INTERFACE
    -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes)
  if(MPILAB_WERROR)
    target_compile_options(mpilab_warnings INTERFACE -Werror)
  endif()
endif()

# ------------------------------------------------------------------ detection
# Done once here so an example never calls find_package itself; each one just
# reads MPILAB_HAVE_* and links the imported targets.
set(MPILAB_HAVE_OPENMP OFF)
set(MPILAB_HAVE_MPI    OFF)
set(MPILAB_HAVE_CUDA   OFF)

if(MPILAB_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS C)
  if(OpenMP_C_FOUND)
    set(MPILAB_HAVE_OPENMP ON)
  endif()
endif()

if(MPILAB_ENABLE_MPI)
  find_package(MPI COMPONENTS C)
  if(MPI_C_FOUND)
    set(MPILAB_HAVE_MPI ON)
  endif()
endif()

if(MPILAB_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # Covers Pascal through Ada; `native` would be faster to compile but
      # breaks cross-compiling on a login node without a GPU.
      set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
    endif()
    set(MPILAB_HAVE_CUDA ON)
  endif()
endif()

if(MPILAB_BUILD_TESTS)
  enable_testing()
endif()

include(MpiLabExample)

# ------------------------------------------------------------------- examples
# Ordered simplest first: this list is the learning path described in the README.
add_subdirectory(examples/hello-mpi)
add_subdirectory(examples/ping-pong)
add_subdirectory(examples/distributed-sum)
add_subdirectory(examples/matrix-multiply)
add_subdirectory(examples/hybrid-object-matching)

mpilab_write_example_map(${CMAKE_BINARY_DIR}/examples.map)

# ------------------------------------------------------------------- summary
message(STATUS "")
message(STATUS "mpi-local-lab ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP     : ${MPILAB_HAVE_OPENMP}")
message(STATUS "  MPI        : ${MPILAB_HAVE_MPI}")
message(STATUS "  CUDA       : ${MPILAB_HAVE_CUDA}")
message(STATUS "  binaries   : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
