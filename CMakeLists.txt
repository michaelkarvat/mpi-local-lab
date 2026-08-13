cmake_minimum_required(VERSION 3.18)
project(hybrid-parallel-object-matching
        VERSION 1.0.0
        DESCRIPTION "Hybrid MPI + OpenMP + CUDA submatrix search"
        LANGUAGES C)

# --------------------------------------------------------------------------
# The build detects MPI, OpenMP and CUDA rather than requiring them. The serial
# backend has no dependencies, so `cmake . && cmake --build .` produces a
# working binary on a bare laptop; the same tree gains backends automatically
# on a cluster. Every optional feature can also be forced off, which is what
# lets CI test each configuration in isolation.
# --------------------------------------------------------------------------
option(MSEARCH_ENABLE_MPI    "Build the MPI runtime if MPI is found"    ON)
option(MSEARCH_ENABLE_OPENMP "Build the OpenMP backend if it is found"  ON)
option(MSEARCH_ENABLE_CUDA   "Build the CUDA backend if it is found"    ON)
option(MSEARCH_BUILD_TESTS   "Build the test suite"                     ON)
option(MSEARCH_WERROR        "Treat warnings as errors"                 OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(msearch_warnings INTERFACE)
if(MSVC)
  target_compile_options(msearch_warnings INTERFACE /W4)
  if(MSEARCH_WERROR)
    target_compile_options(msearch_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(msearch_warnings INTERFACE
    -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes)
  if(MSEARCH_WERROR)
    target_compile_options(msearch_warnings INTERFACE -Werror)
  endif()
endif()

# ------------------------------------------------------------------ features
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

if(MSEARCH_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS C)
  if(OpenMP_C_FOUND)
    list(APPEND MSEARCH_SOURCES src/backends/backend_openmp.c)
    list(APPEND MSEARCH_DEFINES MSEARCH_HAVE_OPENMP)
    list(APPEND MSEARCH_LIBS OpenMP::OpenMP_C)
  endif()
endif()

if(MSEARCH_ENABLE_MPI)
  find_package(MPI COMPONENTS C)
  if(MPI_C_FOUND)
    list(APPEND MSEARCH_SOURCES src/runtime/runner_mpi.c)
    list(APPEND MSEARCH_DEFINES MSEARCH_HAVE_MPI)
    list(APPEND MSEARCH_LIBS MPI::MPI_C)
  endif()
endif()

if(MSEARCH_ENABLE_CUDA)
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
    list(APPEND MSEARCH_SOURCES src/backends/backend_cuda.cu)
    list(APPEND MSEARCH_DEFINES MSEARCH_HAVE_CUDA)
  endif()
endif()

# ------------------------------------------------------------------- targets
add_library(msearch_core STATIC ${MSEARCH_SOURCES})
target_include_directories(msearch_core PUBLIC include PRIVATE src)
target_compile_definitions(msearch_core PUBLIC ${MSEARCH_DEFINES}
  PRIVATE MSEARCH_VERSION="${PROJECT_VERSION}")
target_link_libraries(msearch_core PUBLIC ${MSEARCH_LIBS} PRIVATE msearch_warnings)
if(NOT MSVC)
  target_link_libraries(msearch_core PUBLIC m)
endif()

add_executable(msearch src/cli/main.c)
target_link_libraries(msearch PRIVATE msearch_core msearch_warnings)

if(MSEARCH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ------------------------------------------------------------------- summary
message(STATUS "")
message(STATUS "hybrid-parallel-object-matching ${PROJECT_VERSION}")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP     : ${OpenMP_C_FOUND}")
message(STATUS "  MPI        : ${MPI_C_FOUND}")
message(STATUS "  CUDA       : ${CMAKE_CUDA_COMPILER}")
message(STATUS "")
