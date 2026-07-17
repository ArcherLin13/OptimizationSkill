cmake_minimum_required(VERSION 3.16)
project(planar_rgb_to_cvmat LANGUAGES CXX)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(OPENCL_HEADERS_DIR "${ROOT}/../ocr_softmax_bench/third_party/OpenCL-Headers" CACHE PATH
    "Khronos OpenCL-Headers")

# CPU/NEON ROI crop test — no OpenCL / OpenCV / dma_buf
add_executable(test_roi_crop
    device_test/test_roi_crop.cpp
    device_test/neon_planar_to_cv32fc3.cpp
)
target_compile_features(test_roi_crop PRIVATE cxx_std_17)
target_include_directories(test_roi_crop PRIVATE "${ROOT}/device_test")
if(OHOS OR CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    target_link_libraries(test_roi_crop PRIVATE pthread)
endif()

if(EXISTS "${OPENCL_HEADERS_DIR}/CL/cl.h")
    set(OCL_PLANAR_SRCS
        device_test/ocl_test_planar.cpp
        device_test/neon_planar_to_cv32fc3.cpp
    )
    if(EXISTS "${ROOT}/device_test/dma_buf_cl_import.cpp")
        list(APPEND OCL_PLANAR_SRCS device_test/dma_buf_cl_import.cpp)
    else()
        message(WARNING "dma_buf_cl_import.cpp missing — ocl_test_planar without DMA-BUF helper")
    endif()

    add_executable(ocl_test_planar ${OCL_PLANAR_SRCS})
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
