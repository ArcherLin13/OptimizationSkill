// HarmonyOS MindSpore Lite: load vs infer parallelism benchmark (HiSilicon / NNRT).
//
// Tests:
//   1. serial_load_then_infer  — load model B, then infer model A
//   2. overlap_load_and_infer  — thread loads B while main thread infers A
//   3. infer_parallel_off/on   — OH_AI_ContextSetEnableParallel false vs true
//
// Usage:
//   ./ms_bench --model testdata/tiny.ms [--device nnrt|cpu] [--runs 10]

#include <mindspore/context.h>
#include <mindspore/model.h>
#include <mindspore/status.h>
#include <mindspore/tensor.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Args {
    std::string model_path = "testdata/tiny.ms";
    std::string device = "nnrt";  // nnrt | cpu
    int runs = 10;
    int warmup = 2;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            a.model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            a.device = argv[++i];
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            a.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: %s --model testdata/tiny.ms [--device nnrt|cpu] [--runs N]\n"
                "  nnrt = NNRT NPU + CPU heterogeneous (default)\n"
                "  cpu  = CPU only\n",
                argv[0]);
            std::exit(0);
        }
    }
    return a;
}

const char* statusName(OH_AI_Status st) {
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

void checkStatus(OH_AI_Status st, const char* msg) {
    if (st != OH_AI_STATUS_SUCCESS) {
        std::fprintf(stderr, "MindSpore error %d (%s): %s\n", static_cast<int>(st), statusName(st), msg);
        std::exit(1);
    }
}

bool fileExistsWithSize(const char* path, long long* out_bytes) {
    struct stat st {};
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        return false;
    }
    if (out_bytes) {
        *out_bytes = static_cast<long long>(st.st_size);
    }
    return true;
}

OH_AI_ContextHandle makeContext(const std::string& device, bool enable_parallel) {
    OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
    OH_AI_ContextSetThreadNum(ctx, 4);
    OH_AI_ContextSetEnableParallel(ctx, enable_parallel);

    if (device == "cpu") {
        OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
        OH_AI_ContextAddDeviceInfo(ctx, cpu);
        return ctx;
    }

    // Official HarmonyOS pattern: NNRT accelerator + CPU fallback for unsupported ops.
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
        std::fprintf(stderr, "  using NNRT accelerator + CPU fallback\n");
    } else {
        std::fprintf(stderr, "  NNRT accelerator not found, CPU only\n");
    }

    OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    OH_AI_ContextAddDeviceInfo(ctx, cpu);
    return ctx;
}

struct LoadedModel {
    OH_AI_ContextHandle ctx = nullptr;
    OH_AI_ModelHandle model = nullptr;
};

void destroyLoadedModel(LoadedModel& bundle) {
    if (bundle.model) {
        OH_AI_ModelDestroy(&bundle.model);
        bundle.model = nullptr;
    }
    if (bundle.ctx) {
        OH_AI_ContextDestroy(&bundle.ctx);
        bundle.ctx = nullptr;
    }
}

LoadedModel buildModel(const char* path, const std::string& device, bool enable_parallel) {
    LoadedModel bundle;
    long long bytes = 0;
    if (!fileExistsWithSize(path, &bytes)) {
        std::fprintf(stderr, "Model file missing or unreadable: %s\n", path);
        return bundle;
    }
    std::fprintf(stderr, "  loading %s (%lld bytes)\n", path, bytes);

    bundle.ctx = makeContext(device, enable_parallel);
    bundle.model = OH_AI_ModelCreate();
    const OH_AI_Status st =
        OH_AI_ModelBuildFromFile(bundle.model, path, OH_AI_MODELTYPE_MINDIR, bundle.ctx);
    if (st != OH_AI_STATUS_SUCCESS) {
        std::fprintf(stderr, "ModelBuildFromFile failed (%d %s): %s\n", static_cast<int>(st), statusName(st),
                     path);
        destroyLoadedModel(bundle);
        return bundle;
    }
    return bundle;
}

void zeroInputTensors(OH_AI_TensorHandleArray inputs) {
    for (size_t i = 0; i < inputs.handle_num; ++i) {
        OH_AI_TensorHandle tensor = inputs.handle_list[i];
        void* data = OH_AI_TensorGetMutableData(tensor);
        const size_t nbytes = OH_AI_TensorGetDataSize(tensor);
        if (data && nbytes > 0) {
            std::memset(data, 0, nbytes);
        }
    }
}

double predictOnce(OH_AI_ModelHandle model) {
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    OH_AI_TensorHandleArray outputs = OH_AI_ModelGetOutputs(model);
    zeroInputTensors(inputs);
    const auto t0 = Clock::now();
    checkStatus(OH_AI_ModelPredict(model, inputs, &outputs, nullptr, nullptr), "predict");
    return msSince(t0);
}

double benchPredict(OH_AI_ModelHandle model, int warmup, int runs) {
    for (int i = 0; i < warmup; ++i) {
        (void)predictOnce(model);
    }
    double sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        sum += predictOnce(model);
    }
    return sum / runs;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    std::printf("HiSilicon / MindSpore Lite load-infer parallelism bench\n");
    std::printf("  model: %s\n", args.model_path.c_str());
    std::printf("  device: %s\n", args.device.c_str());
    std::printf("  runs=%d warmup=%d\n\n", args.runs, args.warmup);

    const auto t_build_a = Clock::now();
    LoadedModel model_a = buildModel(args.model_path.c_str(), args.device, false);
    if (!model_a.model) {
        std::fprintf(stderr, "\nTry: ./ms_bench --device cpu --model %s\n", args.model_path.c_str());
        return 1;
    }
    const double load_a_ms = msSince(t_build_a);
    std::printf("Model A ready (first load): %.2f ms\n\n", load_a_ms);

    // --- 1. Serial: load B then infer A ---
    destroyLoadedModel(model_a);
    model_a = buildModel(args.model_path.c_str(), args.device, false);
    if (!model_a.model) {
        return 1;
    }

    const auto t_serial0 = Clock::now();
    LoadedModel model_b = buildModel(args.model_path.c_str(), args.device, false);
    const double serial_load_b_ms = msSince(t_serial0);
    if (!model_b.model) {
        destroyLoadedModel(model_a);
        return 1;
    }

    const double serial_infer_ms = benchPredict(model_a.model, args.warmup, args.runs);
    const double serial_total_ms = serial_load_b_ms + serial_infer_ms;
    destroyLoadedModel(model_b);

    // --- 2. Overlap: load B on worker thread while infer A ---
    destroyLoadedModel(model_a);
    model_a = buildModel(args.model_path.c_str(), args.device, false);
    if (!model_a.model) {
        return 1;
    }

    std::atomic<bool> load_ok{true};
    double overlap_load_ms = 0.0;

    const auto t_overlap0 = Clock::now();
    std::thread loader([&] {
        const auto t0 = Clock::now();
        LoadedModel tmp = buildModel(args.model_path.c_str(), args.device, false);
        overlap_load_ms = msSince(t0);
        if (!tmp.model) {
            load_ok = false;
        }
        destroyLoadedModel(tmp);
    });

    for (int i = 0; i < args.warmup; ++i) {
        (void)predictOnce(model_a.model);
    }
    double overlap_infer_sum = 0.0;
    for (int i = 0; i < args.runs; ++i) {
        overlap_infer_sum += predictOnce(model_a.model);
    }
    loader.join();
    const double overlap_wall_ms = msSince(t_overlap0);
    const double overlap_infer_ms = overlap_infer_sum / args.runs;

    // --- 3. EnableParallel off vs on (infer only) ---
    destroyLoadedModel(model_a);

    LoadedModel par_off = buildModel(args.model_path.c_str(), args.device, false);
    if (!par_off.model) {
        return 1;
    }
    const double infer_par_off = benchPredict(par_off.model, args.warmup, args.runs);
    destroyLoadedModel(par_off);

    LoadedModel par_on = buildModel(args.model_path.c_str(), args.device, true);
    if (!par_on.model) {
        return 1;
    }
    const double infer_par_on = benchPredict(par_on.model, args.warmup, args.runs);

    // --- Report ---
    std::printf("=== load + infer (model B load while A already running) ===\n");
    std::printf("  serial:  load_B=%.2f ms + infer_A=%.2f ms  => total=%.2f ms\n", serial_load_b_ms,
                serial_infer_ms, serial_total_ms);
    std::printf("  overlap: wall=%.2f ms  (load_B=%.2f ms, infer_A=%.2f ms parallel)\n", overlap_wall_ms,
                overlap_load_ms, overlap_infer_ms);
    if (load_ok) {
        const double saved = serial_total_ms - overlap_wall_ms;
        std::printf("  overlap saves ~%.2f ms vs serial (%.2fx)\n", saved,
                    serial_total_ms / std::max(overlap_wall_ms, 0.001));
    } else {
        std::printf("  overlap load FAILED\n");
    }

    std::printf("\n=== OH_AI_ContextSetEnableParallel (infer only) ===\n");
    std::printf("  parallel=false: %.3f ms\n", infer_par_off);
    std::printf("  parallel=true:  %.3f ms\n", infer_par_on);
    std::printf("  speedup: %.2fx\n", infer_par_off / std::max(infer_par_on, 0.001));

    std::printf("\nNotes:\n");
    std::printf("  - overlap: OS threads; NPU may still serialize load+infer internally.\n");
    std::printf("  - EnableParallel: MindSpore op-level parallelism inside one model.\n");
    std::printf("  - If overlap wall ~ max(load,infer), HW allows parallel load+execute.\n");
    std::printf("  - If overlap wall ~ load+infer, they are serialized on device.\n");

    destroyLoadedModel(par_on);
    return load_ok ? 0 : 1;
}
