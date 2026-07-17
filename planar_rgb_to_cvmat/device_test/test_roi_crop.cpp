// HarmonyOS: compare full-frame trans2cv vs 16x ROI crop+trans.
// Scenario default: 4096x3072, 16 boxes ~1000x150.
//
// Usage:
//   ./test_roi_crop [--width 4096] [--height 3072] [--boxes 16] [--box-w 1000] [--box-h 150]

#include "neon_planar_to_cv32fc3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
                "Usage: %s [--width W] [--height H] [--boxes N] [--box-w BW] [--box-h BH] "
                "[--runs R]\n"
                "  Default: 4096x3072, 16 boxes of 1000x150\n",
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

void cropRoiToBgr(const float* r, const float* g, const float* b, int src_stride, const Rect& roi,
                  float* dst_bgr) {
    const float* r0 = r + roi.y * src_stride + roi.x;
    const float* g0 = g + roi.y * src_stride + roi.x;
    const float* b0 = b + roi.y * src_stride + roi.x;
    neon_planar_rgb_f32_to_cv32fc3(r0, g0, b0, dst_bgr, roi.w, roi.h, src_stride, roi.w * 3);
}

void fullThenCropRef(const float* full_bgr, int full_w, const Rect& roi, float* dst_bgr) {
    for (int y = 0; y < roi.h; ++y) {
        const float* src = full_bgr + ((roi.y + y) * full_w + roi.x) * 3;
        float* dst = dst_bgr + y * roi.w * 3;
        std::memcpy(dst, src, static_cast<size_t>(roi.w) * 3 * sizeof(float));
    }
}

float maxAbsDiff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
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

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    const int W = args.width;
    const int H = args.height;
    const int stride = W;

    std::printf("=== ROI vs full trans2cv (CPU/NEON) ===\n");
    std::printf("NEON: %s\n", neon_planar_available() ? "yes" : "scalar fallback");
    std::printf("full image: %d x %d  (%.2f MP)\n", W, H, W * H / 1e6);
    std::printf("boxes: %d x (%d x %d) = %.2f MP total\n", args.boxes, args.box_w, args.box_h,
                args.boxes * args.box_w * args.box_h / 1e6);
    std::printf("pixel ratio (boxes/full): %.1f%%\n\n",
                100.0 * args.boxes * args.box_w * args.box_h / (static_cast<double>(W) * H));

    std::vector<float> r, g, b;
    fillPlanes(r, g, b, W, H);
    const std::vector<Rect> boxes = makeBoxes(args);

    std::vector<float> full_bgr(static_cast<size_t>(W) * H * 3);
    neon_planar_rgb_f32_to_cv32fc3(r.data(), g.data(), b.data(), full_bgr.data(), W, H, stride,
                                   W * 3);

    // Correctness on first few boxes
    bool all_ok = true;
    std::printf("--- correctness (ROI crop == full-trans then crop) ---\n");
    const int check_n = std::min(4, static_cast<int>(boxes.size()));
    for (int i = 0; i < check_n; ++i) {
        const Rect& roi = boxes[i];
        std::vector<float> crop(static_cast<size_t>(roi.w) * roi.h * 3);
        std::vector<float> ref(crop.size());
        cropRoiToBgr(r.data(), g.data(), b.data(), stride, roi, crop.data());
        fullThenCropRef(full_bgr.data(), W, roi, ref.data());
        const float diff = maxAbsDiff(crop.data(), ref.data(), crop.size());
        const bool ok = diff <= 1e-5f;
        all_ok = all_ok && ok;
        std::printf("  box[%d] (%d,%d,%d,%d) max_diff=%.3e %s\n", i, roi.x, roi.y, roi.w, roi.h,
                    diff, ok ? "OK" : "FAIL");
    }

    // Preallocate crop buffers for timing (exclude alloc)
    std::vector<std::vector<float>> crops(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        crops[i].resize(static_cast<size_t>(boxes[i].w) * boxes[i].h * 3);
    }

    const double full_ms = benchMs(args.warmup, args.runs, [&] {
        neon_planar_rgb_f32_to_cv32fc3(r.data(), g.data(), b.data(), full_bgr.data(), W, H, stride,
                                       W * 3);
    });

    const double rois_ms = benchMs(args.warmup, args.runs, [&] {
        for (size_t i = 0; i < boxes.size(); ++i) {
            cropRoiToBgr(r.data(), g.data(), b.data(), stride, boxes[i], crops[i].data());
        }
    });

    std::printf("\n--- convert duration only (avg over %d runs) ---\n", args.runs);
    std::printf("  full trans once:           %.3f ms\n", full_ms);
    std::printf("  %d ROI trans (sum):         %.3f ms\n", args.boxes, rois_ms);
    std::printf("  speedup (full / rois):     %.2fx\n", full_ms / std::max(rois_ms, 1e-9));
    std::printf("  per-box avg:               %.3f ms\n", rois_ms / args.boxes);

    std::printf("\nApp usage:\n");
    std::printf("  #include \"planar_roi_crop.h\"\n");
    std::printf("  cv::Mat crop = planar::cropAabbFromBuffers(r,g,b,W,H,W, roi);\n");

    return all_ok ? 0 : 1;
}
