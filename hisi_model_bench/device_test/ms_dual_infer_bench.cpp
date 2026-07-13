// Compare sequential vs parallel inference of two NPU models on HarmonyOS (NNRT).
//
// Models A and B are loaded once; benchmark measures infer only:
//   serial:   predict(A) then predict(B)
//   parallel: predict(A) and predict(B) on two threads (wall clock)
//
// Usage:
//   ./ms_dual_bench --model-a testdata/tiny.ms [--model-b path] [--device nnrt] [--runs 20]

#include "ms_common.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

struct Args {
    std::string model_a = "testdata/mobilenetv2.ms";
    std::string model_b;
    std::string device = "nnrt";
    int runs = 20;
    int warmup = 3;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model-a") == 0 && i + 1 < argc) {
            a.model_a = argv[++i];
        } else if (std::strcmp(argv[i], "--model-b") == 0 && i + 1 < argc) {
            a.model_b = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            a.model_a = argv[++i];
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            a.device = argv[++i];
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            a.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: %s --model-a testdata/mobilenetv2.ms [--model-b path] [--device nnrt|cpu] [--runs N]\n"
                "  If --model-b omitted, same file as --model-a (two independent instances).\n",
                argv[0]);
            std::exit(0);
        }
    }
    if (a.model_b.empty()) {
        a.model_b = a.model_a;
    }
    return a;
}

struct PairTiming {
    double wall_ms = 0.0;
    double a_ms = 0.0;
    double b_ms = 0.0;
};

PairTiming runSerial(ms::LoadedModel& a, ms::LoadedModel& b, int runs) {
    PairTiming acc{};
    for (int i = 0; i < runs; ++i) {
        const auto t0 = ms::Clock::now();
        const double ta = ms::predictOnceMs(a.model);
        const double tb = ms::predictOnceMs(b.model);
        acc.wall_ms += ms::msSince(t0);
        acc.a_ms += ta;
        acc.b_ms += tb;
    }
    acc.wall_ms /= runs;
    acc.a_ms /= runs;
    acc.b_ms /= runs;
    return acc;
}

PairTiming runParallel(ms::LoadedModel& a, ms::LoadedModel& b, int runs, std::atomic<bool>& ok) {
    PairTiming acc{};
    for (int i = 0; i < runs; ++i) {
        double ta = 0.0;
        double tb = 0.0;
        const auto t0 = ms::Clock::now();

        std::thread worker([&] {
            const auto t_b = ms::Clock::now();
            if (ms::predictOnce(b.model) != OH_AI_STATUS_SUCCESS) {
                ok = false;
            }
            tb = ms::msSince(t_b);
        });

        const auto t_a = ms::Clock::now();
        if (ms::predictOnce(a.model) != OH_AI_STATUS_SUCCESS) {
            ok = false;
        }
        ta = ms::msSince(t_a);
        worker.join();

        acc.wall_ms += ms::msSince(t0);
        acc.a_ms += ta;
        acc.b_ms += tb;
    }
    acc.wall_ms /= runs;
    acc.a_ms /= runs;
    acc.b_ms /= runs;
    return acc;
}

void warmupModels(ms::LoadedModel& a, ms::LoadedModel& b, int n) {
    for (int i = 0; i < n; ++i) {
        (void)ms::predictOnce(a.model);
        (void)ms::predictOnce(b.model);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    std::printf("NPU dual-model infer bench (serial vs parallel)\n");
    std::printf("  model A: %s\n", args.model_a.c_str());
    std::printf("  model B: %s\n", args.model_b.c_str());
    std::printf("  device: %s\n", args.device.c_str());
    std::printf("  runs=%d warmup=%d\n\n", args.runs, args.warmup);

    std::fprintf(stderr, "Loading model A...\n");
    ms::LoadedModel model_a = ms::buildModel(args.model_a.c_str(), args.device);
    if (!model_a.model) {
        return 1;
    }
    std::fprintf(stderr, "Loading model B...\n");
    ms::LoadedModel model_b = ms::buildModel(args.model_b.c_str(), args.device, true);
    if (!model_b.model) {
        ms::destroyLoadedModel(model_a);
        return 1;
    }

    std::printf("Load time:  A=%.2f ms  B=%.2f ms\n\n", model_a.load_ms, model_b.load_ms);

    warmupModels(model_a, model_b, args.warmup);

    const PairTiming serial = runSerial(model_a, model_b, args.runs);

    std::atomic<bool> parallel_ok{true};
    const PairTiming parallel = runParallel(model_a, model_b, args.runs, parallel_ok);

    std::printf("=== infer only (models pre-loaded) ===\n");
    std::printf("  serial wall:    %.3f ms  (A=%.3f + B=%.3f)\n", serial.wall_ms, serial.a_ms,
                serial.b_ms);
    std::printf("  parallel wall:  %.3f ms  (A=%.3f, B=%.3f concurrent)\n", parallel.wall_ms,
                parallel.a_ms, parallel.b_ms);

    const double saved = serial.wall_ms - parallel.wall_ms;
    const double speedup = serial.wall_ms / std::max(parallel.wall_ms, 1e-6);
    std::printf("\n=== comparison ===\n");
    std::printf("  saved:   %.3f ms\n", saved);
    std::printf("  speedup: %.2fx\n", speedup);

    if (parallel.wall_ms >= serial.wall_ms * 0.95) {
        std::printf("  => NPU likely serializes two models (wall ~ sum of both)\n");
    } else if (parallel.wall_ms <= std::max(serial.a_ms, serial.b_ms) * 1.15) {
        std::printf("  => NPU may run two models in parallel (wall ~ max(A,B))\n");
    } else {
        std::printf("  => partial overlap (wall between max and sum)\n");
    }

    std::printf("\nNotes:\n");
    std::printf("  - Each model uses its own Context (MindSpore API requirement).\n");
    std::printf("  - parallel = two OS threads calling ModelPredict at the same time.\n");
    std::printf("  - Use two different --model-a/--model-b paths for real dual-model workloads.\n");

    ms::destroyLoadedModel(model_a);
    ms::destroyLoadedModel(model_b);
    return parallel_ok ? 0 : 1;
}
