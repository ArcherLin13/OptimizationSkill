cmake_minimum_required(VERSION 3.16)
project(hisi_model_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED OHOS_NATIVE)
    set(OHOS_NATIVE "C:/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/native")
endif()
set(MS_INCLUDE "${OHOS_NATIVE}/sysroot/usr/include")
set(MS_LIBDIR "${OHOS_NATIVE}/sysroot/usr/lib/aarch64-linux-ohos")

add_executable(ms_bench device_test/ms_load_infer_bench.cpp)
target_include_directories(ms_bench PRIVATE "${MS_INCLUDE}")
target_link_directories(ms_bench PRIVATE "${MS_LIBDIR}")
target_link_libraries(ms_bench PRIVATE mindspore_lite_ndk pthread)
