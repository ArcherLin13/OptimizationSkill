cmake_minimum_required(VERSION 3.16)
project(seamless_clone_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED OpenCV_DIR AND DEFINED ENV{OPENCV_OHOS_DIR})
  set(OpenCV_DIR "$ENV{OPENCV_OHOS_DIR}/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc photo)

add_executable(seamless_clone_bench
  src/main.cpp
  src/benchmark.cpp
  src/optimized_clone.cpp
)

target_include_directories(seamless_clone_bench PRIVATE src)
target_link_libraries(seamless_clone_bench PRIVATE opencv_core opencv_imgproc opencv_photo)

if(OHOS)
  set_target_properties(seamless_clone_bench PROPERTIES
    OUTPUT_NAME "seamless_clone_bench"
  )
endif()
