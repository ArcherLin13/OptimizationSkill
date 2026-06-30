cmake_minimum_required(VERSION 3.16)
project(__BENCH_NAME__ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BENCH_EXE __BENCH_NAME__)
set(OPENCV_DEVICE_DIR "${CMAKE_SOURCE_DIR}/opencv" CACHE PATH "Device OpenCV libs + headers")

option(USE_DEVICE_OPENCV "Link OpenCV from OPENCV_DEVICE_DIR" ON)

add_executable(${BENCH_EXE}
  src/main.cpp
  src/bench_case.cpp
  src/baseline.cpp
  src/optimized.cpp
  src/benchmark.cpp
  lib/your_optimized/your_optimized.cpp
)

target_include_directories(${BENCH_EXE} PRIVATE
  src
  framework
  lib/your_optimized
)

if(USE_DEVICE_OPENCV)
  foreach(_inc IN ITEMS "${OPENCV_DEVICE_DIR}/include" "${OPENCV_DEVICE_DIR}/include/opencv4")
    if(EXISTS "${_inc}")
      target_include_directories(${BENCH_EXE} PRIVATE "${_inc}")
    endif()
  endforeach()
  foreach(_mod IN ITEMS core imgproc)
    file(GLOB _libs "${OPENCV_DEVICE_DIR}/libopencv_${_mod}.so*" "${OPENCV_DEVICE_DIR}/lib/libopencv_${_mod}.so*")
    if(_libs)
      list(GET _libs 0 _lib)
      target_link_libraries(${BENCH_EXE} PRIVATE "${_lib}")
    else()
      message(FATAL_ERROR "Missing libopencv_${_mod}.so under ${OPENCV_DEVICE_DIR}")
    endif()
  endforeach()
endif()
