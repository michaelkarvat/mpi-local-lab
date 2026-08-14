# Run one example at several rank counts and assert something about the result.
#
# Generalised from the object-matching example's mpi_equivalence.cmake, which
# proved a specific version of this for msearch. The property it checks is the
# one that makes a distributed program safe to change: the number of ranks must
# not be observable in the answer.
#
# Invoked by mpilab_add_rank_test(); not meant to be run by hand.
#
#   BINARY                the example executable
#   MPIEXEC               mpirun/mpiexec
#   MPIEXEC_NUMPROC_FLAG  usually -n
#   RANKS                 comma-separated, e.g. "1,2,4"
#   MODE                  invariant | ranks-present
#   ARGS                  comma-separated extra arguments, may be empty
#   WORKDIR               directory to run in and write scratch output to
#
# MODE=invariant       every rank count must produce byte-identical output.
#                      Use for anything with a defined answer.
# MODE=ranks-present   at N ranks the output must carry exactly one line per
#                      rank, each announcing a distinct rank id in 0..N-1. Use
#                      for programs whose lines legitimately arrive in any
#                      order, where the point is that every rank ran.

if(NOT DEFINED MODE OR MODE STREQUAL "")
  set(MODE "invariant")
endif()

string(REPLACE "," ";" rank_list "${RANKS}")
set(extra_args "")
if(DEFINED ARGS AND NOT ARGS STREQUAL "")
  string(REPLACE "," ";" extra_args "${ARGS}")
endif()

get_filename_component(binary_name "${BINARY}" NAME_WE)

# Launch, retrying without --oversubscribe for launchers that reject it. Some
# MPICH builds do; OpenMPI needs it whenever ranks exceed the visible cores,
# which is the normal case on a laptop and inside a container.
function(run_at_ranks ranks out_file out_status)
  execute_process(
    COMMAND "${MPIEXEC}" ${MPIEXEC_NUMPROC_FLAG} ${ranks} --oversubscribe
            "${BINARY}" ${extra_args}
    OUTPUT_FILE "${out_file}"
    ERROR_VARIABLE err
    RESULT_VARIABLE code)
  if(NOT code EQUAL 0)
    execute_process(
      COMMAND "${MPIEXEC}" ${MPIEXEC_NUMPROC_FLAG} ${ranks} "${BINARY}" ${extra_args}
      OUTPUT_FILE "${out_file}"
      ERROR_VARIABLE err
      RESULT_VARIABLE code)
  endif()
  if(NOT code EQUAL 0)
    message(FATAL_ERROR "${binary_name}: mpirun at ${ranks} rank(s) failed with ${code}\n${err}")
  endif()
  set(${out_status} "${code}" PARENT_SCOPE)
endfunction()

set(baseline "")

foreach(ranks IN LISTS rank_list)
  set(actual "${WORKDIR}/${binary_name}.${ranks}.out")
  run_at_ranks(${ranks} "${actual}" code)

  if(MODE STREQUAL "invariant")
    if(baseline STREQUAL "")
      set(baseline "${actual}")
      message(STATUS "${binary_name}: ${ranks} rank(s) -> baseline")
    else()
      execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${baseline}" "${actual}"
                      RESULT_VARIABLE differs)
      if(NOT differs EQUAL 0)
        file(READ "${baseline}" want)
        file(READ "${actual}" got)
        message(FATAL_ERROR
          "${binary_name}: output at ${ranks} rank(s) differs from the baseline.\n"
          "--- baseline ---\n${want}\n--- ${ranks} ranks ---\n${got}")
      endif()
      message(STATUS "${binary_name}: ${ranks} rank(s) -> identical to baseline")
    endif()

  elseif(MODE STREQUAL "ranks-present")
    file(STRINGS "${actual}" lines)
    list(LENGTH lines line_count)
    if(NOT line_count EQUAL ranks)
      file(READ "${actual}" got)
      message(FATAL_ERROR
        "${binary_name}: expected ${ranks} line(s) at ${ranks} rank(s), got ${line_count}.\n${got}")
    endif()
    # Each rank must announce itself exactly once. Order is not asserted:
    # stdout from N processes interleaves arbitrarily and pinning that would be
    # testing the MPI implementation's buffering, not this program.
    math(EXPR last "${ranks} - 1")
    foreach(r RANGE ${last})
      set(seen 0)
      foreach(line IN LISTS lines)
        if(line MATCHES "rank ${r} of ${ranks}")
          math(EXPR seen "${seen} + 1")
        endif()
      endforeach()
      if(NOT seen EQUAL 1)
        file(READ "${actual}" got)
        message(FATAL_ERROR
          "${binary_name}: rank ${r} of ${ranks} appeared ${seen} time(s), expected 1.\n${got}")
      endif()
    endforeach()
    message(STATUS "${binary_name}: ${ranks} rank(s) -> all ranks reported once")

  else()
    message(FATAL_ERROR "RunRanks.cmake: unknown MODE '${MODE}'")
  endif()
endforeach()
