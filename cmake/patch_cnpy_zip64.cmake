if(NOT DEFINED SOURCE_DIR)
	message(FATAL_ERROR "SOURCE_DIR not set")
endif()

if(NOT DEFINED PATCH_FILE)
	message(FATAL_ERROR "PATCH_FILE not set")
endif()

set(cnpy_cpp "${SOURCE_DIR}/cnpy.cpp")
if(NOT EXISTS "${cnpy_cpp}")
	message(FATAL_ERROR "cnpy.cpp not found at: ${cnpy_cpp}")
endif()

file(READ "${cnpy_cpp}" cnpy_cpp_content)
if(cnpy_cpp_content MATCHES "parse_zip64_sizes")
	message(STATUS "cnpy already patched for ZIP64 support")
	return()
endif()

find_program(PATCH_EXECUTABLE patch)
if(NOT PATCH_EXECUTABLE)
	message(FATAL_ERROR "Required tool 'patch' not found in PATH")
endif()

execute_process(
	COMMAND "${PATCH_EXECUTABLE}" -p1 -i "${PATCH_FILE}"
	WORKING_DIRECTORY "${SOURCE_DIR}"
	RESULT_VARIABLE patch_result
	OUTPUT_VARIABLE patch_out
	ERROR_VARIABLE patch_err
)

if(NOT patch_result EQUAL 0)
	message(FATAL_ERROR "Failed to apply cnpy ZIP64 patch (exit=${patch_result})\n${patch_out}\n${patch_err}")
endif()

message(STATUS "Applied cnpy ZIP64 patch")
