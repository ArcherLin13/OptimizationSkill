#include "bench_case.h"
#include "log_format.h"
#include "metrics.h"
#include "optimized_clone.h"
#include "poisson_jacobi.h"
#include "seamless_roi.h"
#include "timing.h"

#include <functional>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace {

constexpr int kWarmupRuns = 2;
constexpr int kMeasureRuns = 5;
constexpr int kJacobiIterations = 400;

enum class VariantKind { Reference, Identical, Fast, Approx };

const char* kindLabel(VariantKind kind) {
    switch (kind) {
        case VariantKind::Reference:
            return "REF";
        case VariantKind::Identical:
            return "SAME";
        case VariantKind::Fast:
            return "FAST";
        case VariantKind::Approx:
            return "APRX";
    }
    return "?";
}

BenchCase makeUserCase() {
    return makeVisualBenchCase();
}

struct Variant {
    std::string name;
    VariantKind kind;
    std::function<void(cv::Mat&)> run;
    double psnrThreshold;
    double maxDiffThreshold;
    const char* note;
};

struct VariantResult {
    Variant variant;
    TimingStats stats{};
    CompareResult compare{};
    bool hasCompare = false;
};

TimingStats timeVariant(const Variant& variant) {
    return measureMs(kWarmupRuns, kMeasureRuns, [&]() {
        cv::Mat out;
        variant.run(out);
    });
}

void printLegend() {
    printSubBanner("列说明");
    std::cout << "  kind   REF=参考基准  SAME=应与baseline逐像素一致  FAST=更快但结果不同  APRX=近似算法\n";
    std::cout << "  ok     PASS=达到该路径的质量阈值  FAIL=未达到\n";
    std::cout << "  计时   " << kWarmupRuns << " 次预热 + " << kMeasureRuns << " 次采样，取 avg/min/max\n";
}

}  // namespace

int runBenchmark() {
    printBanner("seamlessClone Benchmark (NORMAL_CLONE)");

    const BenchCase bench = makeUserCase();
    const int maskFg = cv::countNonZero(bench.mask);
    const int solverPx = solverBoundingRectPixels(bench);

    printSubBanner("环境");
    printKv("OpenCV", CV_VERSION);
    printKv("CPU threads", std::to_string(cv::getNumThreads()));
    printKv("OpenCL", isOpenCLPoissonAvailable() ? "yes" : "no");

    printSubBanner("测试用例");
    printKv("src / dst", std::to_string(bench.src.cols) + "x" + std::to_string(bench.src.rows) +
                            "  CV_8UC3");
    printKv("mask", "rect(5,5,718,115)  fg=" + std::to_string(maskFg));
    printKv("center", "(" + std::to_string(bench.center.x) + ", " + std::to_string(bench.center.y) +
                          ")");
    printKv("solver bbox", std::to_string(solverPx) + " px  (OpenCV boundingRect(mask))");
    printKv("Jacobi iters", std::to_string(kJacobiIterations));

    cv::Mat baselineOut;
    runBaselineClone(bench, baselineOut);

    PooledCloneContext pool;
    pool.srcBuf.create(bench.src.size(), bench.src.type());
    pool.dstBuf.create(bench.dst.size(), bench.dst.type());
    pool.maskBuf.create(bench.mask.size(), bench.mask.type());
    pool.outBuf.create(bench.dst.size(), bench.dst.type());

    cv::Mat preallocOut;
    preallocOut.create(bench.dst.size(), bench.dst.type());

    std::vector<Variant> variants;
    variants.push_back({"baseline", VariantKind::Reference,
                         [&](cv::Mat& out) { runBaselineClone(bench, out); }, 100.0, 0.0,
                         "矩形 mask，OpenCV 默认路径"});
    variants.push_back(
        {"prealloc_out", VariantKind::Identical,
         [&](cv::Mat& out) {
             runPreallocOutClone(bench, preallocOut);
             out = preallocOut;
         },
         100.0, 0.0, "预先 create 输出 Mat"});
    variants.push_back({"pooled_reuse", VariantKind::Identical,
                        [&](cv::Mat& out) { runPooledClone(pool, bench, out); }, 100.0, 0.0,
                        "复用 src/dst/mask/out buffer"});

    for (int threads : {1, 2, 4, 8}) {
        variants.push_back({"threads_" + std::to_string(threads), VariantKind::Identical,
                            [&, threads](cv::Mat& out) { runBaselineCloneThreads(bench, out, threads); },
                            100.0, 0.0, "cv::setNumThreads 扫描"});
    }

    variants.push_back({"full_mask", VariantKind::Fast,
                        [&](cv::Mat& out) { runFullMaskClone(bench, out); }, 42.0, 6.0,
                        "mask 全 255，solver bbox 更大但 FFT 更快"});
    variants.push_back({"full_mask_border_paste5", VariantKind::Fast,
                        [&](cv::Mat& out) { runFullMaskBorderPasteClone(bench, out, 5); }, 42.0, 6.0,
                        "满 mask clone 后贴回 5px 边框"});
    variants.push_back({"aligned_736x128", VariantKind::Fast,
                        [&](cv::Mat& out) { runAlignedClone(bench, out, 32); }, 42.0, 6.0,
                        "pad 到 736x128"});
    variants.push_back({"half_res", VariantKind::Fast,
                        [&](cv::Mat& out) { runHalfResClone(bench, out); }, 28.0, 25.0,
                        "半分辨率求解后放大"});
    variants.push_back({"jacobi_cpu", VariantKind::Approx,
                        [&](cv::Mat& out) { runJacobiPoissonClone(bench, out, kJacobiIterations, false); },
                        28.0, 25.0, "CPU Jacobi Poisson（GPU 同类算法）"});

    if (isOpenCLPoissonAvailable()) {
        variants.push_back({"jacobi_opencl", VariantKind::Approx,
                            [&](cv::Mat& out) {
                                runJacobiPoissonClone(bench, out, kJacobiIterations, true);
                            },
                            28.0, 25.0, "OpenCL Jacobi Poisson"});
    }

    std::vector<VariantResult> results;
    results.reserve(variants.size());

    double baselineAvgMs = 0.0;
    for (const Variant& v : variants) {
        VariantResult row;
        row.variant = v;
        row.stats = timeVariant(v);

        if (v.name != "baseline") {
            cv::Mat out;
            v.run(out);
            row.compare = compareImages(baselineOut, out, v.psnrThreshold, v.maxDiffThreshold);
            row.hasCompare = true;
        }

        if (v.name == "baseline") {
            baselineAvgMs = row.stats.avgMs;
        }
        results.push_back(row);
    }

    printLegend();
    printSubBanner("结果总表（baseline = 矩形 mask）");
    printResultTableHeader();

    bool identicalPass = true;
    for (const VariantResult& row : results) {
        const double speedup =
            (row.variant.name == "baseline")
                ? 1.0
                : baselineAvgMs / std::max(row.stats.avgMs, 1e-6);

        printResultRow(row.variant.name.c_str(), kindLabel(row.variant.kind), row.stats, speedup,
                       row.hasCompare ? &row.compare : nullptr, row.hasCompare);

        if (row.variant.kind == VariantKind::Identical && row.hasCompare && !row.compare.pass) {
            identicalPass = false;
        }
    }

    printSubBanner("路径说明");
    for (const VariantResult& row : results) {
        if (row.variant.name == "baseline") {
            continue;
        }
        printNote(row.variant.name.c_str(), row.variant.note);
    }

    printSubBanner("补充：仅改 mask 为满 mask");
    cv::Mat fullOut;
    double fullMs = 0.0;
    timeOnce([&]() { runFullMaskClone(bench, fullOut); }, fullMs);
    const CompareResult fullVsRect = compareImages(baselineOut, fullOut, 42.0, 6.0);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  full_mask (同 src/dst):  " << fullMs << " ms"
              << "   vs baseline " << baselineAvgMs << " ms"
              << "   speedup " << (baselineAvgMs / std::max(fullMs, 1e-6)) << "x\n";
    std::cout << "  quality vs rect mask:    PSNR=" << fullVsRect.psnr
              << " dB   maxDiff=" << fullVsRect.maxAbsDiff << "   "
              << (fullVsRect.pass ? "PASS" : "FAIL") << "\n";

    printBanner(identicalPass ? "SAME paths: PASS" : "SAME paths: FAIL");
    std::cout << "  SAME 类路径要求 maxDiff=0（prealloc / pooled / threads）\n\n";

    return identicalPass ? 0 : 1;
}
