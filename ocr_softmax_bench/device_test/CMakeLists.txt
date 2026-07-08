cmake_minimum_required(VERSION 3.16)
project(ocr_softmax_device_test LANGUAGES CXX)

find_package(OpenCL QUIET)

if(NOT OpenCL_FOUND)
    message(WARNING "OpenCL not found — only ocl_test_2d target skipped. Compile manually with -lOpenCL.")
    return()
endif()

add_executable(ocl_test_2d ocl_test_2d.cpp)
target_compile_features(ocl_test_2d PRIVATE cxx_std_17)
target_include_directories(ocl_test_2d PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(ocl_test_2d PRIVATE ${OpenCL_LIBRARIES})
