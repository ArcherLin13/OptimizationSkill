#pragma once

#include <mindspore/context.h>
#include <mindspore/model.h>
#include <mindspore/status.h>
#include <mindspore/tensor.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace ms {

using Clock = std::chrono::steady_clock;

inline double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

inline const char* statusName(OH_AI_Status st) {
    switch (st) {
        case OH_AI_STATUS_SUCCESS:
            return "SUCCESS";
        case OH_AI_STATUS_LITE_ERROR:
            return "LITE_ERROR";
        case OH_AI_STATUS_LITE_PARAM_INVALID:
            return "LITE_PARAM_INVALID";
        case OH_AI_STATUS_LITE_GRAPH_FILE_ERROR:
            return "LITE_GRAPH_FILE_ERROR";
        case OH_AI_STATUS_LITE_NOT_FIND_OP:
            return "LITE_NOT_FIND_OP";
        case OH_AI_STATUS_LITE_NOT_SUPPORT:
            return "LITE_NOT_SUPPORT";
        default:
            return "UNKNOWN";
    }
}

inline bool fileExistsWithSize(const char* path, long long* out_bytes) {
    struct stat st {};
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        return false;
    }
    if (out_bytes) {
        *out_bytes = static_cast<long long>(st.st_size);
    }
    return true;
}

inline void printNnrtDevices() {
    size_t n = 0;
    NNRTDeviceDesc* descs = OH_AI_GetAllNNRTDeviceDescs(&n);
    if (!descs || n == 0) {
        std::fprintf(stderr, "  NNRT devices: none\n");
        return;
    }
    std::fprintf(stderr, "  NNRT devices (%zu):\n", n);
    for (size_t i = 0; i < n; ++i) {
        NNRTDeviceDesc* d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
        auto ty = OH_AI_GetTypeFromNNRTDeviceDesc(d);
        const char* name = OH_AI_GetNameFromNNRTDeviceDesc(d);
        const size_t dev_id = OH_AI_GetDeviceIdFromNNRTDeviceDesc(d);
        std::fprintf(stderr, "    [%zu] %s id=%zu type=%d\n", i, name ? name : "?", dev_id,
                     static_cast<int>(ty));
    }
    OH_AI_DestroyAllNNRTDeviceDescs(&descs);
}

inline void printLoadHints(const char* path, const std::string& device) {
    std::fprintf(stderr, "\nModelBuildFromFile troubleshooting:\n");
    std::fprintf(stderr, "  1. Use official mobilenetv2.ms (HarmonyOS quick_start model)\n");
    std::fprintf(stderr, "  2. Push SDK lib: scripts push libmindspore_lite_ndk.so + LD_LIBRARY_PATH\n");
    std::fprintf(stderr, "  3. Try CPU only: --device cpu\n");
    std::fprintf(stderr, "  4. tiny/add.ms (1 KB) often fails on newer MindSpore builds\n");
    std::fprintf(stderr, "  path=%s device=%s\n\n", path, device.c_str());
    if (device == "nnrt") {
        printNnrtDevices();
    }
}

inline OH_AI_ContextHandle makeContext(const std::string& device) {
    OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
    OH_AI_ContextSetThreadNum(ctx, 4);
    OH_AI_ContextSetEnableParallel(ctx, false);

    if (device == "cpu") {
        OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
        OH_AI_ContextAddDeviceInfo(ctx, cpu);
        std::fprintf(stderr, "  context: CPU only\n");
        return ctx;
    }

    std::fprintf(stderr, "  context: NNRT + CPU fallback\n");
    OH_AI_DeviceInfoHandle nnrt = OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDEVICE_ACCELERATOR);
    if (!nnrt) {
        size_t n = 0;
        NNRTDeviceDesc* descs = OH_AI_GetAllNNRTDeviceDescs(&n);
        if (descs && n > 0) {
            for (size_t i = 0; i < n; ++i) {
                NNRTDeviceDesc* d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
                auto ty = OH_AI_GetTypeFromNNRTDeviceDesc(d);
                const char* name = OH_AI_GetNameFromNNRTDeviceDesc(d);
                std::fprintf(stderr, "  NNRT[%zu]: %s type=%d\n", i, name ? name : "?", static_cast<int>(ty));
                if (ty == OH_AI_NNRTDEVICE_ACCELERATOR && !nnrt && name) {
                    nnrt = OH_AI_CreateNNRTDeviceInfoByName(name);
                }
            }
            OH_AI_DestroyAllNNRTDeviceDescs(&descs);
        }
    }
    if (nnrt) {
        OH_AI_DeviceInfoSetPerformanceMode(nnrt, OH_AI_PERFORMANCE_HIGH);
        OH_AI_ContextAddDeviceInfo(ctx, nnrt);
    }

    OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    OH_AI_ContextAddDeviceInfo(ctx, cpu);
    return ctx;
}

struct LoadedModel {
    OH_AI_ContextHandle ctx = nullptr;
    OH_AI_ModelHandle model = nullptr;
    double load_ms = 0.0;
};

inline void destroyLoadedModel(LoadedModel& bundle) {
    if (bundle.model) {
        OH_AI_ModelDestroy(&bundle.model);
        bundle.model = nullptr;
    }
    if (bundle.ctx) {
        OH_AI_ContextDestroy(&bundle.ctx);
        bundle.ctx = nullptr;
    }
}

inline LoadedModel buildModel(const char* path, const std::string& device, bool quiet = false) {
    LoadedModel bundle;
    long long bytes = 0;
    if (!fileExistsWithSize(path, &bytes)) {
        std::fprintf(stderr, "Model file missing: %s\n", path);
        return bundle;
    }
    if (!quiet) {
        std::fprintf(stderr, "  loading %s (%lld bytes)\n", path, bytes);
    }

    const auto t0 = Clock::now();
    bundle.ctx = makeContext(device);
    bundle.model = OH_AI_ModelCreate();
    const OH_AI_Status st =
        OH_AI_ModelBuildFromFile(bundle.model, path, OH_AI_MODELTYPE_MINDIR, bundle.ctx);
    bundle.load_ms = msSince(t0);
    if (st != OH_AI_STATUS_SUCCESS) {
        std::fprintf(stderr, "ModelBuildFromFile failed (%d %s): %s\n", static_cast<int>(st),
                     statusName(st), path);
        printLoadHints(path, device);
        destroyLoadedModel(bundle);
        return bundle;
    }
    return bundle;
}

inline void zeroInputTensors(OH_AI_TensorHandleArray inputs) {
    for (size_t i = 0; i < inputs.handle_num; ++i) {
        OH_AI_TensorHandle tensor = inputs.handle_list[i];
        void* data = OH_AI_TensorGetMutableData(tensor);
        const size_t nbytes = OH_AI_TensorGetDataSize(tensor);
        if (data && nbytes > 0) {
            std::memset(data, 0, nbytes);
        }
    }
}

inline OH_AI_Status predictOnce(OH_AI_ModelHandle model) {
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    OH_AI_TensorHandleArray outputs = OH_AI_ModelGetOutputs(model);
    zeroInputTensors(inputs);
    return OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr);
}

inline double predictOnceMs(OH_AI_ModelHandle model) {
    const auto t0 = Clock::now();
    (void)predictOnce(model);
    return msSince(t0);
}

}  // namespace ms
