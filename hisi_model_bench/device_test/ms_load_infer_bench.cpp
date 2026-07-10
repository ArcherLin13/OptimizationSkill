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
                "  nnrt = HiSilicon NPU via NNRT (default)\n"
                "  cpu  = CPU only\n",
                argv[0]);
            std::exit(0);
        }
    }
    return a;
}

void checkStatus(OH_AI_Status st, const char* msg) {
    if (st != OH_AI_STATUS_SUCCESS) {
        std::fprintf(stderr, "MindSpore error %d: %s\n", static_cast<int>(st), msg);
        std::exit(1);
    }
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

    // NNRT: prefer HiSilicon accelerator (NPU)
    size_t n = 0;
    NNRTDeviceDesc* descs = OH_AI_GetAllNNRTDeviceDescs(&n);
    OH_AI_DeviceInfoHandle dev = nullptr;
    if (descs && n > 0) {
        for (size_t i = 0; i < n; ++i) {
            NNRTDeviceDesc* d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
            auto ty = OH_AI_GetTypeFromNNRTDeviceDesc(d);
            const char* name = OH_AI_GetNameFromNNRTDeviceDesc(d);
            std::fprintf(stderr, "  NNRT[%zu]: %s type=%d\n", i, name ? name : "?", static_cast<int>(ty));
            if (ty == OH_AI_NNRTDEVICE_ACCELERATOR && !dev) {
                dev = OH_AI_CreateNNRTDeviceInfoByName(name);
            }
        }
        OH_AI_DestroyAllNNRTDeviceDescs(&descs);
    }
    if (!dev) {
        std::fprintf(stderr, "NNRT accelerator not found, fallback CPU\n");
        dev = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    } else {
        OH_AI_DeviceInfoSetPerformanceMode(dev, OH_AI_PERFORMANCE_HIGH);
    }
    OH_AI_ContextAddDeviceInfo(ctx, dev);
    return ctx;
}

OH_AI_ModelHandle buildModel(const char* path, OH_AI_ContextHandle ctx) {
    OH_AI_ModelHandle model = OH_AI_ModelCreate();
    OH_AI_Status st =
        OH_AI_ModelBuildFromFile(model, path, OH_AI_MODELTYPE_MINDIR, ctx);
    if (st != OH_AI_STATUS_SUCCESS) {
        std::fprintf(stderr, "ModelBuildFromFile failed (%d): %s\n", static_cast<int>(st), path);
        OH_AI_ModelDestroy(&model);
        return nullptr;
    }
    return model;
}

void destroyModel(OH_AI_ModelHandle& model) {
    if (model) {
        OH_AI_ModelDestroy(&model);
        model = nullptr;
    }
}

double predictOnce(OH_AI_ModelHandle model) {
    OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
    OH_AI_TensorHandleArray outputs = OH_AI_ModelGetOutputs(model);
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

struct TimedResult {
    const char* name;
    double ms;
    const char* note;
};

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    std::printf("HiSilicon / MindSpore Lite load-infer parallelism bench\n");
    std::printf("  model: %s\n", args.model_path.c_str());
    std::printf("  device: %s\n", args.device.c_str());
    std::printf("  runs=%d warmup=%d\n\n", args.runs, args.warmup);

    OH_AI_ContextHandle ctx = makeContext(args.device, false);
    const auto t_build_a = Clock::now();
    OH_AI_ModelHandle model_a = buildModel(args.model_path.c_str(), ctx);
    if (!model_a) {
        return 1;
    }
    const double load_a_ms = msSince(t_build_a);
    std::printf("Model A ready (first load): %.2f ms\n\n", load_a_ms);

    // --- 1. Serial: load B then infer A ---
    destroyModel(model_a);
    model_a = buildModel(args.model_path.c_str(), ctx);

    const auto t_serial0 = Clock::now();
    OH_AI_ModelHandle model_b = buildModel(args.model_path.c_str(), ctx);
    const double serial_load_b_ms = msSince(t_serial0);

    const double serial_infer_ms = benchPredict(model_a, args.warmup, args.runs);
    const double serial_total_ms = serial_load_b_ms + serial_infer_ms;
    destroyModel(model_b);

    // --- 2. Overlap: load B on worker thread while infer A ---
    model_a = buildModel(args.model_path.c_str(), ctx);
    std::atomic<bool> load_ok{true};
    double overlap_load_ms = 0.0;

    const auto t_overlap0 = Clock::now();
    std::thread loader([&] {
        const auto t0 = Clock::now();
        OH_AI_ModelHandle tmp = buildModel(args.model_path.c_str(), ctx);
        overlap_load_ms = msSince(t0);
        if (!tmp) {
            load_ok = false;
        }
        destroyModel(tmp);
    });

  for (int i = 0; i < args.warmup; ++i) {
        (void)predictOnce(model_a);
    }
    double overlap_infer_sum = 0.0;
    for (int i = 0; i < args.runs; ++i) {
        overlap_infer_sum += predictOnce(model_a);
    }
    loader.join();
    const double overlap_wall_ms = msSince(t_overlap0);
    const double overlap_infer_ms = overlap_infer_sum / args.runs;

    // --- 3. EnableParallel off vs on (infer only) ---
    destroyModel(model_a);
    OH_AI_ContextDestroy(&ctx);

    ctx = makeContext(args.device, false);
    model_a = buildModel(args.model_path.c_str(), ctx);
    const double infer_par_off = benchPredict(model_a, args.warmup, args.runs);

    destroyModel(model_a);
    OH_AI_ContextDestroy(&ctx);

    ctx = makeContext(args.device, true);
    model_a = buildModel(args.model_path.c_str(), ctx);
    const double infer_par_on = benchPredict(model_a, args.warmup, args.runs);

    // --- Report ---
    std::printf("=== load + infer (model B load while A already running) ===\n");
    std::printf("  serial:  load_B=%.2f ms + infer_A=%.2f ms  => total=%.2f ms\n", serial_load_b_ms,
                serial_infer_ms, serial_total_ms);
    std::printf("  overlap: wall=%.2f ms  (load_B=%.2f ms, infer_A=%.2f ms parallel)\n",
                overlap_wall_ms, overlap_load_ms, overlap_infer_ms);
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

    destroyModel(model_a);
    OH_AI_ContextDestroy(&ctx);
    return load_ok ? 0 : 1;
}
