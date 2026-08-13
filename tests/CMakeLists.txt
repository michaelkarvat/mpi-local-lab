function(msearch_add_test name)
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PRIVATE msearch_core msearch_warnings)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND ${name})
  # Tests write scratch files into the directory they run in.
  set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

msearch_add_test(test_metric)
msearch_add_test(test_parser)
msearch_add_test(test_options)
msearch_add_test(test_backends)

# End-to-end: the CLI on the shipped fixtures, checked against golden output.
add_test(NAME cli_golden_output
         COMMAND ${CMAKE_COMMAND}
                 -DMSEARCH_BIN=$<TARGET_FILE:msearch>
                 -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_golden_output.cmake)

# The MPI runtime must produce byte-identical output to a single process, for
# every rank count. This is the property that makes the distributed runtime
# safe to change.
if(MPI_C_FOUND AND MPIEXEC_EXECUTABLE)
  add_test(NAME mpi_equivalence
           COMMAND ${CMAKE_COMMAND}
                   -DMSEARCH_BIN=$<TARGET_FILE:msearch>
                   -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data
                   -DMPIEXEC=${MPIEXEC_EXECUTABLE}
                   -DMPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/mpi_equivalence.cmake)
endif()
