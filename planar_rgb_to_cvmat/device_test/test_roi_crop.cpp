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
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
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
                "  full vs ROI (1t), then ROI thread sweep + when_all\n",
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

// Persistent pool: T workers pull box indices (atomic). Spawn cost outside timed region.
class BoxWorkerPool {
public:
    BoxWorkerPool(int threads, const float* r, const float* g, const float* b, int stride,
                  const std::vector<Rect>* boxes, std::vector<std::vector<float>>* crops)
        : threads_(std::max(1, threads)),
          r_(r),
          g_(g),
          b_(b),
          stride_(stride),
          boxes_(boxes),
          crops_(crops) {
        workers_.reserve(static_cast<size_t>(threads_));
        for (int t = 0; t < threads_; ++t) {
            workers_.emplace_back([this] { workerLoop(); });
        }
    }

    ~BoxWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++epoch_;
        }
        cv_.notify_all();
        for (auto& th : workers_) {
            th.join();
        }
    }

    void runAll() {
        const int n = static_cast<int>(boxes_->size());
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mu_);
            ++epoch_;
            active_ = true;
        }
        cv_.notify_all();

        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [&] { return done_.load(std::memory_order_acquire) >= n; });
        active_ = false;
    }

private:
    void workerLoop() {
        int seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || (active_ && epoch_ != seen); });
            if (stop_) {
                return;
            }
            seen = epoch_;
            lock.unlock();

            const int n = static_cast<int>(boxes_->size());
            for (;;) {
                const int i = next_.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) {
                    break;
                }
                transRoi(r_, g_, b_, stride_, (*boxes_)[static_cast<size_t>(i)],
                         (*crops_)[static_cast<size_t>(i)].data());
                if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 >= n) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    int threads_ = 1;
    const float* r_ = nullptr;
    const float* g_ = nullptr;
    const float* b_ = nullptr;
    int stride_ = 0;
    const std::vector<Rect>* boxes_ = nullptr;
    std::vector<std::vector<float>>* crops_ = nullptr;

    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    bool stop_ = false;
    bool active_ = false;
    int epoch_ = 0;
    std::atomic<int> next_{0};
    std::atomic<int> done_{0};
    std::vector<std::thread> workers_;
};

// when_all: one std::thread per box, join all (includes spawn/join cost).
void runWhenAll(const float* r, const float* g, const float* b, int stride,
                const std::vector<Rect>& boxes, std::vector<std::vector<float>>& crops) {
    std::vector<std::thread> workers;
    workers.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        workers.emplace_back([&, i] {
            transRoi(r, g, b, stride, boxes[i], crops[i].data());
        });
    }
    for (auto& th : workers) {
        th.join();
    }
}

std::vector<int> threadSweep(int max_t) {
    std::vector<int> out;
    for (int t = 1; t <= max_t; ++t) {
        if (t == 1 || t == 2 || t == 4 || t == 6 || t == 8 || t == max_t || (t % 4 == 0 && t <= 16)) {
            if (out.empty() || out.back() != t) {
                out.push_back(t);
            }
        }
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
        for (size_t i = 0; i < boxes.size(); ++i) {
            transRoi(r.data(), g.data(), b.data(), W, boxes[i], crops[i].data());
        }
    });

    std::printf("--- CPU 1-thread convert time (avg %d runs) ---\n", args.runs);
    std::printf("  full image trans once:     %.3f ms\n", full_ms);
    std::printf("  %d box ROI trans (sum):     %.3f ms\n", args.boxes, boxes_1t_ms);
    std::printf("  per box avg:               %.3f ms\n", boxes_1t_ms / args.boxes);
    std::printf("  full / boxes_1t:           %.2fx\n\n", full_ms / std::max(boxes_1t_ms, 1e-9));

    const int max_pool = std::min(args.boxes, static_cast<int>(hw));
    const std::vector<int> sweep = threadSweep(max_pool);

    std::printf("--- box ROI MT (persistent pool, convert only) ---\n");
    std::printf("  %-10s %10s %10s %12s\n", "threads", "ms", "vs_1t", "vs_full");
    double best_ms = boxes_1t_ms;
    int best_t = 1;
    for (int t : sweep) {
        BoxWorkerPool pool(t, r.data(), g.data(), b.data(), W, &boxes, &crops);
        // warm pool once before timed runs
        pool.runAll();
        const double ms = benchMs(args.warmup, args.runs, [&] { pool.runAll(); });
        std::printf("  pool%-5d  %10.3f %9.2fx %11.2fx\n", t, ms, boxes_1t_ms / std::max(ms, 1e-9),
                    full_ms / std::max(ms, 1e-9));
        if (ms < best_ms) {
            best_ms = ms;
            best_t = t;
        }
    }

    const double when_all_ms = benchMs(args.warmup, args.runs, [&] {
        runWhenAll(r.data(), g.data(), b.data(), W, boxes, crops);
    });
    std::printf("  when_all   %10.3f %9.2fx %11.2fx  (spawn %d threads/run)\n", when_all_ms,
                boxes_1t_ms / std::max(when_all_ms, 1e-9), full_ms / std::max(when_all_ms, 1e-9),
                args.boxes);

    std::printf("\n  best pool: %d threads (%.3f ms)\n", best_t, best_ms);
    if (when_all_ms <= best_ms * 1.08) {
        std::printf("  => when_all ≈ best pool; OK to fire all boxes if N is small (~%d)\n",
                    args.boxes);
    } else {
        std::printf("  => prefer pool(~%d); when_all pays spawn/oversubscribe\n", best_t);
    }
    return 0;
}
