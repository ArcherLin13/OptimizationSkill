cmake_minimum_required(VERSION 3.16)
project(planar_rgb_to_cvmat LANGUAGES CXX)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(OPENCL_HEADERS_DIR "${ROOT}/../ocr_softmax_bench/third_party/OpenCL-Headers" CACHE PATH
    "Khronos OpenCL-Headers")

if(EXISTS "${OPENCL_HEADERS_DIR}/CL/cl.h")
    add_executable(ocl_test_planar
        device_test/ocl_test_planar.cpp
        device_test/neon_planar_to_cv32fc3.cpp
    )
    target_compile_features(ocl_test_planar PRIVATE cxx_std_17)
    target_include_directories(ocl_test_planar PRIVATE "${OPENCL_HEADERS_DIR}" "${ROOT}/device_test")
    target_compile_definitions(ocl_test_planar PRIVATE CL_TARGET_OPENCL_VERSION=120)

    if(OHOS OR CMAKE_SYSTEM_NAME STREQUAL "OHOS")
        target_sources(ocl_test_planar PRIVATE device_test/opencl_dynload.cpp)
        target_compile_definitions(ocl_test_planar PRIVATE OCR_OPENCL_DLOPEN=1)
        target_link_libraries(ocl_test_planar PRIVATE dl pthread)
        message(STATUS "ocl_test_planar: OHOS dynamic OpenCL (dlopen libOpenCL.so)")
    elseif(DEFINED OPENCL_LIBRARY AND OPENCL_LIBRARY)
        target_link_libraries(ocl_test_planar PRIVATE "${OPENCL_LIBRARY}")
    else()
        target_link_libraries(ocl_test_planar PRIVATE OpenCL)
    endif()
else()
    message(WARNING "OpenCL headers missing at ${OPENCL_HEADERS_DIR}/CL/cl.h")
endif()
