cmake_minimum_required(VERSION 3.16)
project(planar_rgb_to_cvmat LANGUAGES CXX)

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# CPU 1-thread: full trans vs box ROI trans (no OpenCL / OpenCV)
add_executable(test_roi_crop
    device_test/test_roi_crop.cpp
    device_test/neon_planar_to_cv32fc3.cpp
)
target_compile_features(test_roi_crop PRIVATE cxx_std_17)
target_include_directories(test_roi_crop PRIVATE "${ROOT}/device_test")
if(OHOS OR CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    target_link_libraries(test_roi_crop PRIVATE pthread)
endif()
