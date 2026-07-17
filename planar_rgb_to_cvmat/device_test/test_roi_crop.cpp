// CPU: full-image trans2cv vs N box ROI trans2cv.
// Also sweeps box-parallel thread counts vs when_all (1 thread per box).
// Default: 4096x3072, 16 boxes of 1000x150.
//
// Usage:
//   ./test_roi_crop [--width 4096] [--height 3072] [--boxes 16] [--box-w 1000] [--box-h 150]

#include "neon_planar_to_cv32fc3.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
};

struct Args {
    int width = 4096;
    int height = 3072;
    int boxes = 16;
    int box_w = 1000;
    int box_h = 150;
    int runs = 30;
    int warmup = 5;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            a.width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            a.height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--boxes") == 0 && i + 1 < argc) {
            a.boxes = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--box-w") == 0 && i + 1 < argc) {
            a.box_w = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--box-h") == 0 && i + 1 < argc) {
            a.box_h = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            a.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: %s [--width W] [--height H] [--boxes N] [--box-w BW] [--box-h BH]\n"
                "  full vs ROI (1t), then ROI mt sweep + when_all\n",
                argv[0]);
            std::exit(0);
        }
    }
    return a;
}

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void fillPlanes(std::vector<float>& r, std::vector<float>& g, std::vector<float>& b, int w, int h) {
    const size_t n = static_cast<size_t>(w) * h;
    r.resize(n);
    g.resize(n);
    b.resize(n);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            r[i] = static_cast<float>((x * 13 + y * 7) % 256) / 255.f;
            g[i] = static_cast<float>((x * 3 + y * 17) % 256) / 255.f;
            b[i] = static_cast<float>((x * 29 + y * 5) % 256) / 255.f;
        }
    }
}

std::vector<Rect> makeBoxes(const Args& a) {
    std::vector<Rect> out;
    out.reserve(static_cast<size_t>(a.boxes));
    const int bw = std::min(a.box_w, a.width);
    const int bh = std::min(a.box_h, a.height);
    const int cols = 4;
    const int rows = (a.boxes + cols - 1) / cols;
    for (int i = 0; i < a.boxes; ++i) {
        const int col = i % cols;
        const int row = i / cols;
        int x = (a.width - bw) * (col + 1) / (cols + 1);
        int y = (a.height - bh) * (row + 1) / (rows + 1);
        x = std::max(0, std::min(x, a.width - bw));
        y = std::max(0, std::min(y, a.height - bh));
        out.push_back({x, y, bw, bh});
    }
    return out;
}

void transRoi(const float* r, const float* g, const float* b, int stride, const Rect& roi,
              float* dst_bgr) {
    neon_planar_rgb_f32_to_cv32fc3(r + roi.y * stride + roi.x, g + roi.y * stride + roi.x,
                                   b + roi.y * stride + roi.x, dst_bgr, roi.w, roi.h, stride,
                                   roi.w * 3);
}

void transFull(const float* r, const float* g, const float* b, int w, int h, float* dst_bgr) {
    neon_planar_rgb_f32_to_cv32fc3(r, g, b, dst_bgr, w, h, w, w * 3);
}

template <typename Fn>
double benchMs(int warmup, int runs, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    double sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = Clock::now();
        fn();
        sum += msSince(t0);
    }
    return sum / runs;
}

void runBoxesSerial(const float* r, const float* g, const float* b, int stride,
                    const std::vector<Rect>& boxes, std::vector<std::vector<float>>& crops) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        transRoi(r, g, b, stride, boxes[i], crops[i].data());
    }
}

// T worker threads steal boxes via atomic index, then join.
// threads==1 => same path as serial (no spawn). Includes spawn/join when threads>1.
void runBoxesMt(int threads, const float* r, const float* g, const float* b, int stride,
                const std::vector<Rect>& boxes, std::vector<std::vector<float>>& crops) {
    const int n = static_cast<int>(boxes.size());
    threads = std::max(1, std::min(threads, n));
    if (threads == 1) {
        runBoxesSerial(r, g, b, stride, boxes, crops);
        return;
    }

    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&] {
            for (;;) {
                const int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) {
                    break;
                }
                transRoi(r, g, b, stride, boxes[static_cast<size_t>(i)],
                         crops[static_cast<size_t>(i)].data());
            }
        });
    }
    for (auto& th : workers) {
        th.join();
    }
}

std::vector<int> threadSweep(int max_t) {
    std::vector<int> out;
    for (int t : {1, 2, 4, 6, 8, 12, 16}) {
        if (t <= max_t) {
            out.push_back(t);
        }
    }
    if (out.empty() || out.back() != max_t) {
        out.push_back(max_t);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    const int W = args.width;
    const int H = args.height;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    std::printf("=== CPU: full trans vs box ROI (+ MT sweep) ===\n");
    std::printf("impl: %s | hardware_concurrency=%u\n",
                neon_planar_available() ? "NEON" : "scalar", hw);
    std::printf("full:  %d x %d = %.2f MP\n", W, H, W * H / 1e6);
    std::printf("boxes: %d x (%d x %d) = %.2f MP (%.1f%% of full)\n\n", args.boxes, args.box_w,
                args.box_h, args.boxes * args.box_w * args.box_h / 1e6,
                100.0 * args.boxes * args.box_w * args.box_h / (static_cast<double>(W) * H));

    std::vector<float> r, g, b;
    fillPlanes(r, g, b, W, H);
    const std::vector<Rect> boxes = makeBoxes(args);

    std::vector<float> full_bgr(static_cast<size_t>(W) * H * 3);
    std::vector<std::vector<float>> crops(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        crops[i].resize(static_cast<size_t>(boxes[i].w) * boxes[i].h * 3);
    }

    // Correctness: first box ROI == crop from full interleaved
    transFull(r.data(), g.data(), b.data(), W, H, full_bgr.data());
    {
        const Rect& roi = boxes[0];
        transRoi(r.data(), g.data(), b.data(), W, roi, crops[0].data());
        float max_diff = 0.f;
        for (int y = 0; y < roi.h; ++y) {
            const float* ref_row = full_bgr.data() + ((roi.y + y) * W + roi.x) * 3;
            const float* crop_row = crops[0].data() + y * roi.w * 3;
            for (int i = 0; i < roi.w * 3; ++i) {
                max_diff = std::max(max_diff, std::fabs(crop_row[i] - ref_row[i]));
            }
        }
        std::printf("correctness box0 vs full-crop: max_diff=%.3e %s\n\n", max_diff,
                    max_diff <= 1e-5f ? "OK" : "FAIL");
        if (max_diff > 1e-5f) {
            return 1;
        }
    }

    const double full_ms = benchMs(args.warmup, args.runs, [&] {
        transFull(r.data(), g.data(), b.data(), W, H, full_bgr.data());
    });

    const double boxes_1t_ms = benchMs(args.warmup, args.runs, [&] {
        runBoxesSerial(r.data(), g.data(), b.data(), W, boxes, crops);
    });

    std::printf("--- CPU 1-thread convert time (avg %d runs) ---\n", args.runs);
    std::printf("  full image trans once:     %.3f ms\n", full_ms);
    std::printf("  %d box ROI trans (sum):     %.3f ms\n", args.boxes, boxes_1t_ms);
    std::printf("  per box avg:               %.3f ms\n", boxes_1t_ms / args.boxes);
    std::printf("  full / boxes_1t:           %.2fx\n\n", full_ms / std::max(boxes_1t_ms, 1e-9));

    const int max_t = std::min(args.boxes, static_cast<int>(hw));
    const std::vector<int> sweep = threadSweep(max_t);

    std::printf("--- box ROI MT (spawn T threads / run, steal boxes) ---\n");
    std::printf("  %-10s %10s %10s %12s\n", "threads", "ms", "vs_1t", "vs_full");
    double best_ms = boxes_1t_ms;
    int best_t = 1;
    for (int t : sweep) {
        const double ms = benchMs(args.warmup, args.runs, [&] {
            runBoxesMt(t, r.data(), g.data(), b.data(), W, boxes, crops);
        });
        std::printf("  mt%-7d  %10.3f %9.2fx %11.2fx%s\n", t, ms,
                    boxes_1t_ms / std::max(ms, 1e-9), full_ms / std::max(ms, 1e-9),
                    t == 1 ? "  (= serial)" : "");
        if (ms < best_ms) {
            best_ms = ms;
            best_t = t;
        }
    }

    const double when_all_ms = benchMs(args.warmup, args.runs, [&] {
        runBoxesMt(args.boxes, r.data(), g.data(), b.data(), W, boxes, crops);
    });
    std::printf("  when_all   %10.3f %9.2fx %11.2fx  (mt%d = 1 thread/box)\n", when_all_ms,
                boxes_1t_ms / std::max(when_all_ms, 1e-9), full_ms / std::max(when_all_ms, 1e-9),
                args.boxes);

    std::printf("\n  best: mt%d (%.3f ms)\n", best_t, best_ms);
    if (when_all_ms <= best_ms * 1.08) {
        std::printf("  => when_all ≈ best; OK to fire all boxes if N is small (~%d)\n", args.boxes);
    } else {
        std::printf("  => prefer mt%d; when_all pays spawn/oversubscribe\n", best_t);
    }
    return 0;
}
