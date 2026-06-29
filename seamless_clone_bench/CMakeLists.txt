cmake_minimum_required(VERSION 3.16)
project(seamless_clone_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link OpenCV from ./opencv (device libs). Set OFF to use third_party find_package build.
option(USE_DEVICE_OPENCV "Link OpenCV from OPENCV_ROOT folder" ON)
set(OPENCV_ROOT "${CMAKE_SOURCE_DIR}/opencv" CACHE PATH "Folder with device OpenCV libs and headers")

set(OPENCV_DEVICE_MODULES core imgcodecs imgproc photo)

add_executable(seamless_clone_bench
  src/main.cpp
  src/benchmark.cpp
  src/optimized_clone.cpp
)

target_include_directories(seamless_clone_bench PRIVATE src)

function(find_opencv_module_lib out_var module_name)
  set(_search_dirs
    "${OPENCV_ROOT}"
    "${OPENCV_ROOT}/lib"
    "${OPENCV_ROOT}/lib64"
  )
  set(_found_lib "")
  foreach(_dir IN LISTS _search_dirs)
    if(NOT EXISTS "${_dir}")
      continue()
    endif()
    file(GLOB _candidates
      "${_dir}/libopencv_${module_name}.so"
      "${_dir}/libopencv_${module_name}.so.*"
    )
    if(_candidates)
      list(GET _candidates 0 _found_lib)
      break()
    endif()
  endforeach()
  if(NOT _found_lib)
    message(FATAL_ERROR "Missing libopencv_${module_name}.so* under ${OPENCV_ROOT}")
  endif()
  message(STATUS "OpenCV device lib (${module_name}): ${_found_lib}")
  set(${out_var} "${_found_lib}" PARENT_SCOPE)
endfunction()

if(USE_DEVICE_OPENCV)
  set(_include_dirs "")
  foreach(_inc IN ITEMS
      "${OPENCV_ROOT}/include"
      "${OPENCV_ROOT}/include/opencv4"
  )
    if(EXISTS "${_inc}")
      list(APPEND _include_dirs "${_inc}")
    endif()
  endforeach()
  if(NOT _include_dirs)
    message(FATAL_ERROR "OpenCV headers not found under ${OPENCV_ROOT}/include")
  endif()
  target_include_directories(seamless_clone_bench PRIVATE ${_include_dirs})

  foreach(_mod IN LISTS OPENCV_DEVICE_MODULES)
    find_opencv_module_lib(_lib "${_mod}")
    target_link_libraries(seamless_clone_bench PRIVATE "${_lib}")
  endforeach()
else()
  if(NOT DEFINED OpenCV_DIR AND DEFINED ENV{OPENCV_OHOS_DIR})
    set(OpenCV_DIR "$ENV{OPENCV_OHOS_DIR}/lib/cmake/opencv4")
  endif()

  find_package(OpenCV REQUIRED COMPONENTS core imgproc photo)
  target_include_directories(seamless_clone_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(seamless_clone_bench PRIVATE opencv_core opencv_imgproc opencv_photo)
endif()

if(OHOS)
  set_target_properties(seamless_clone_bench PROPERTIES
    OUTPUT_NAME "seamless_clone_bench"
  )
endif()
