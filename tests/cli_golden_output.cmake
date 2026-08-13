# Runs the CLI over every fixture that has a matching .expected file and
# requires byte-identical output. Written in CMake script mode rather than
# shell so the same test runs on Linux, macOS and Windows.
file(GLOB expected_files "${DATA_DIR}/*.expected")
if(expected_files STREQUAL "")
  message(FATAL_ERROR "no fixtures found in ${DATA_DIR}")
endif()

foreach(expected IN LISTS expected_files)
  get_filename_component(stem "${expected}" NAME_WE)
  set(input "${DATA_DIR}/${stem}.txt")
  set(actual "${CMAKE_CURRENT_BINARY_DIR}/${stem}.actual")

  execute_process(
    COMMAND "${MSEARCH_BIN}" --backend serial --quiet -i "${input}" -o "${actual}"
    RESULT_VARIABLE code
    OUTPUT_VARIABLE out
    ERROR_VARIABLE errout)
  if(NOT code EQUAL 0)
    message(FATAL_ERROR "${stem}: msearch exited ${code}\n${out}\n${errout}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files "${expected}" "${actual}"
    RESULT_VARIABLE diff)
  if(NOT diff EQUAL 0)
    file(READ "${expected}" want)
    file(READ "${actual}" got)
    message(FATAL_ERROR "${stem}: output differs\n--- expected ---\n${want}\n--- actual ---\n${got}")
  endif()

  # `-o -` must put results on stdout and nothing else. Timing and logging
  # belong on stderr, or `msearch -o - > file` silently corrupts the results.
  set(piped "${CMAKE_CURRENT_BINARY_DIR}/${stem}.piped")
  execute_process(
    COMMAND "${MSEARCH_BIN}" --backend serial -i "${input}" -o -
    OUTPUT_FILE "${piped}"
    ERROR_VARIABLE ignored
    RESULT_VARIABLE code)
  if(NOT code EQUAL 0)
    message(FATAL_ERROR "${stem}: msearch -o - exited ${code}")
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E compare_files "${expected}" "${piped}"
    RESULT_VARIABLE diff)
  if(NOT diff EQUAL 0)
    file(READ "${piped}" got)
    message(FATAL_ERROR
      "${stem}: stdout from '-o -' is not exactly the results\n--- got ---\n${got}")
  endif()

  message(STATUS "${stem}: ok (file and stdout)")
endforeach()
