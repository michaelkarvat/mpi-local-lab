# Helpers so an example's CMakeLists.txt says what the example needs and
# nothing about how the toolchain was found. Detection already happened once at
# the repository root; these functions only read MPILAB_HAVE_*.
#
# An example that needs something unavailable is skipped with a status line
# rather than failing the configure. That is what lets one `cmake -S . -B build`
# work on a bare laptop, inside the CPU containers, and on a GPU node, which is
# the same graceful-degradation principle the object-matching backends use.

# mpilab_add_example(<name>
#     SOURCES <file>...
#     [MPI] [CUDA]     -- hard requirements; the example is skipped without them
#     [OPENMP]         -- opportunistic; linked when available, absent otherwise
#     [DESCRIPTION <text>])
#
# OPENMP is deliberately not a requirement. An example guards its pragmas with
# #ifdef _OPENMP, so the same source compiles to a single-threaded binary when
# no OpenMP is present -- the compiler defines _OPENMP only under -fopenmp,
# which also keeps -Wunknown-pragmas quiet under -Werror. MPI and CUDA cannot
# degrade that way: without them the program has nothing left to do.
function(mpilab_add_example name)
  cmake_parse_arguments(EX "MPI;OPENMP;CUDA" "DESCRIPTION" "SOURCES" ${ARGN})

  if(NOT EX_SOURCES)
    message(FATAL_ERROR "mpilab_add_example(${name}): SOURCES is required")
  endif()

  foreach(need MPI CUDA)
    if(EX_${need} AND NOT MPILAB_HAVE_${need})
      message(STATUS "example ${name}: skipped, needs ${need}")
      return()
    endif()
  endforeach()

  add_executable(${name} ${EX_SOURCES})
  target_link_libraries(${name} PRIVATE mpilab_warnings)

  if(EX_MPI)
    target_link_libraries(${name} PRIVATE MPI::MPI_C)
    target_compile_definitions(${name} PRIVATE MPILAB_HAVE_MPI)
  endif()
  if(EX_OPENMP AND MPILAB_HAVE_OPENMP)
    target_link_libraries(${name} PRIVATE OpenMP::OpenMP_C)
  endif()
  if(NOT MSVC)
    target_link_libraries(${name} PRIVATE m)
  endif()

  # Record how this example is spelled. scripts/run-mpi.sh takes whatever the
  # docs call the example -- usually the directory -- and needs to turn it into
  # a target to build and a binary to launch. For most examples the three names
  # coincide; hybrid-object-matching builds `msearch`, and guessing is not an
  # option, so the mapping is written down at configure time instead.
  get_filename_component(dir "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  mpilab_register_example("${dir}" "${name}" "${EX_DESCRIPTION}")
endfunction()

# mpilab_register_example(<directory-name> <target> [description])
#
# Called for you by mpilab_add_example(). Call it directly from an example that
# builds its target by hand.
function(mpilab_register_example dir target)
  set_property(GLOBAL APPEND PROPERTY MPILAB_EXAMPLES "${target}")
  set_property(GLOBAL APPEND PROPERTY MPILAB_EXAMPLE_MAP "${dir}=${target}")
  if(NOT "${dir}" STREQUAL "${target}")
    set_property(GLOBAL APPEND PROPERTY MPILAB_EXAMPLE_MAP "${target}=${target}")
  endif()
  set_property(GLOBAL APPEND PROPERTY MPILAB_EXAMPLE_DESC "${target}|${ARGV2}")
endfunction()

# Write the mapping where the in-container scripts can read it. Called once
# from the root, after every example has been added.
function(mpilab_write_example_map path)
  get_property(entries GLOBAL PROPERTY MPILAB_EXAMPLE_MAP)
  set(text "# <name users type>=<cmake target / binary in bin/>\n")
  foreach(entry IN LISTS entries)
    string(APPEND text "${entry}\n")
  endforeach()
  file(WRITE "${path}" "${text}")
endfunction()

# mpilab_add_rank_test(<name>
#     RANKS <n>...                  -- rank counts to sweep, baseline first
#     [MODE invariant|ranks-present] -- default invariant
#     [ARGS <arg>...])
#
# Registers one CTest test that runs the example at every listed rank count.
# Silently does nothing when the example was skipped or no launcher exists, so
# a serial-only configure still produces a green suite.
function(mpilab_add_rank_test name)
  cmake_parse_arguments(RT "" "MODE" "RANKS;ARGS" ${ARGN})

  if(NOT TARGET ${name})
    return()
  endif()
  if(NOT MPILAB_HAVE_MPI OR NOT MPIEXEC_EXECUTABLE)
    return()
  endif()
  if(NOT RT_RANKS)
    set(RT_RANKS 1 2 4)
  endif()
  if(NOT RT_MODE)
    set(RT_MODE invariant)
  endif()

  # Semicolons in a -D value would be split into separate arguments by add_test,
  # so lists travel as comma-separated strings and RunRanks.cmake splits them.
  string(REPLACE ";" "," ranks_csv "${RT_RANKS}")
  string(REPLACE ";" "," args_csv "${RT_ARGS}")

  add_test(NAME ${name}-ranks
           COMMAND ${CMAKE_COMMAND}
                   -DBINARY=$<TARGET_FILE:${name}>
                   -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                   -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                   -DRANKS=${ranks_csv}
                   -DARGS=${args_csv}
                   -DMODE=${RT_MODE}
                   -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}
                   -P ${CMAKE_SOURCE_DIR}/cmake/RunRanks.cmake)
endfunction()
