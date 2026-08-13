# The distributed runtime must be observationally identical to a single
# process: same results, same order, for any number of ranks. Dynamic work
# claiming means ranks process pictures in a nondeterministic *order*, so this
# is a real check that results are reassembled into input order rather than
# written as they arrive.
set(input "${DATA_DIR}/reference.txt")
set(baseline "${CMAKE_CURRENT_BINARY_DIR}/mpi_baseline.out")

execute_process(
  COMMAND "${MSEARCH_BIN}" --backend serial --quiet -i "${input}" -o "${baseline}"
  RESULT_VARIABLE code)
if(NOT code EQUAL 0)
  message(FATAL_ERROR "single-process baseline failed with ${code}")
endif()

foreach(ranks 1 2 3 4)
  set(actual "${CMAKE_CURRENT_BINARY_DIR}/mpi_${ranks}.out")
  execute_process(
    COMMAND "${MPIEXEC}" ${MPIEXEC_NUMPROC_FLAG} ${ranks} --oversubscribe
            "${MSEARCH_BIN}" --backend serial --quiet -i "${input}" -o "${actual}"
    RESULT_VARIABLE code
    ERROR_VARIABLE errout)
  if(NOT code EQUAL 0)
    # Not every launcher understands --oversubscribe; retry without it.
    execute_process(
      COMMAND "${MPIEXEC}" ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              "${MSEARCH_BIN}" --backend serial --quiet -i "${input}" -o "${actual}"
      RESULT_VARIABLE code
      ERROR_VARIABLE errout)
  endif()
  if(NOT code EQUAL 0)
    message(FATAL_ERROR "mpirun -n ${ranks} failed with ${code}\n${errout}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files "${baseline}" "${actual}"
    RESULT_VARIABLE diff)
  if(NOT diff EQUAL 0)
    message(FATAL_ERROR "output with ${ranks} ranks differs from the single-process baseline")
  endif()
  message(STATUS "${ranks} rank(s): identical to baseline")
endforeach()
