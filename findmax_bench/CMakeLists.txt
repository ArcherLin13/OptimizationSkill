cmake_minimum_required(VERSION 3.16)
project(findmax_bench LANGUAGES CXX)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(OPENCL_HEADERS_DIR "${ROOT}/../ocr_softmax_bench/third_party/OpenCL-Headers" CACHE PATH
    "Khronos OpenCL-Headers")

if(NOT EXISTS "${OPENCL_HEADERS_DIR}/CL/cl.h")
    message(FATAL_ERROR "OpenCL headers missing: ${OPENCL_HEADERS_DIR}/CL/cl.h")
endif()

add_executable(ocl_test_findmax device_test/ocl_test_findmax.cpp)
target_compile_features(ocl_test_findmax PRIVATE cxx_std_17)
target_include_directories(ocl_test_findmax PRIVATE "${OPENCL_HEADERS_DIR}" "${ROOT}/device_test")
target_compile_definitions(ocl_test_findmax PRIVATE CL_TARGET_OPENCL_VERSION=120)

if(OHOS OR CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    target_sources(ocl_test_findmax PRIVATE device_test/opencl_dynload.cpp)
    target_compile_definitions(ocl_test_findmax PRIVATE OCR_OPENCL_DLOPEN=1)
    target_link_libraries(ocl_test_findmax PRIVATE dl)
    message(STATUS "ocl_test_findmax: OHOS dynamic OpenCL (dlopen)")
elseif(DEFINED OPENCL_LIBRARY AND OPENCL_LIBRARY)
    target_link_libraries(ocl_test_findmax PRIVATE "${OPENCL_LIBRARY}")
else()
    target_link_libraries(ocl_test_findmax PRIVATE OpenCL)
endif()
