#include "bench_case.h"
#include "log_format.h"
#include "metrics.h"
#include "optimized_clone.h"
#include "poisson_fft.h"
#include "poisson_jacobi.h"
#include "sc_dft_verify.h"
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
    printSubBanner("Legend");
    std::cout << "  kind   REF=reference  SAME=must match baseline exactly  FAST=faster, different result  APRX=approx\n";
    std::cout << "  ok     PASS=meets quality threshold for this path  FAIL=does not meet\n";
    std::cout << "  timing " << kWarmupRuns << " warmup + " << kMeasureRuns << " runs, report avg/min/max\n";
}

}  // namespace

int runBenchmark(const BenchCase& bench) {
    printBanner("seamlessClone Benchmark (NORMAL_CLONE)");

    const int maskFg = cv::countNonZero(bench.mask);
    const int solverPx = solverBoundingRectPixels(bench);

    printSubBanner("Environment");
    printKv("OpenCV", CV_VERSION);
    printKv("CPU threads", std::to_string(cv::getNumThreads()));

    printSubBanner("Test case");
    printKv("src / dst", std::to_string(bench.src.cols) + "x" + std::to_string(bench.src.rows) +
                            "  CV_8UC3");
    printKv("mask", "rect(5,5,718,115)  fg=" + std::to_string(maskFg));
    printKv("center", "(" + std::to_string(bench.center.x) + ", " + std::to_string(bench.center.y) +
                          ")");
    printKv("solver bbox", std::to_string(solverPx) + " px  (OpenCV boundingRect(mask))");
    printKv("Jacobi iters", "400 (jacobi_cpu)");

    sc_fft::logDstDftVerify();

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
                         "rect mask, default OpenCV path"});
    variants.push_back({"poisson_fft", VariantKind::Identical,
                        [&](cv::Mat& out) { runFftPoissonClone(bench, out); }, 100.0, 0.0,
                        "FFT clone, merge+dft DST + parallel_for 3ch"});
    variants.push_back({"poisson_fft_ocv_scalar", VariantKind::Identical,
                        [&](cv::Mat& out) { runFftPoissonNativeClone(bench, out, false); }, 100.0, 0.0,
                        "native OpenCV DFT port (scalar), no NEON"});
    variants.push_back({"poisson_fft_ocv_neon", VariantKind::Identical,
                        [&](cv::Mat& out) { runFftPoissonNativeClone(bench, out, true); }, 100.0, 0.0,
                        "native OpenCV DFT + ARM NEON radix-4"});
    variants.push_back(
        {"prealloc_out", VariantKind::Identical,
         [&](cv::Mat& out) {
             runPreallocOutClone(bench, preallocOut);
             out = preallocOut;
         },
         100.0, 0.0, "pre-create output Mat before clone"});
    variants.push_back({"pooled_reuse", VariantKind::Identical,
                        [&](cv::Mat& out) { runPooledClone(pool, bench, out); }, 100.0, 0.0,
                        "reuse src/dst/mask/out buffers"});

    for (int threads : {1, 2, 4, 8}) {
        variants.push_back({"threads_" + std::to_string(threads), VariantKind::Identical,
                            [&, threads](cv::Mat& out) { runBaselineCloneThreads(bench, out, threads); },
                            100.0, 0.0, "cv::setNumThreads sweep"});
    }

    variants.push_back({"full_mask", VariantKind::Fast,
                        [&](cv::Mat& out) { runFullMaskClone(bench, out); }, 42.0, 6.0,
                        "mask all 255, larger solver bbox but faster FFT"});
    variants.push_back({"full_mask_border_paste5", VariantKind::Fast,
                        [&](cv::Mat& out) { runFullMaskBorderPasteClone(bench, out, 5); }, 42.0, 6.0,
                        "full mask clone, paste back 5px dst border"});
    variants.push_back({"aligned_736x128", VariantKind::Fast,
                        [&](cv::Mat& out) { runAlignedClone(bench, out, 32); }, 42.0, 6.0,
                        "pad to 736x128"});
    variants.push_back({"half_res", VariantKind::Fast,
                        [&](cv::Mat& out) { runHalfResClone(bench, out); }, 28.0, 25.0,
                        "half-res solve then upscale"});
    variants.push_back({"jacobi_cpu", VariantKind::Approx,
                        [&](cv::Mat& out) { runJacobiPoissonClone(bench, out, 400); }, 28.0, 25.0,
                        "CPU Jacobi Poisson, 400 iters"});

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
    printSubBanner("Results (baseline = rect mask)");
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

    printSubBanner("Path notes");
    for (const VariantResult& row : results) {
        if (row.variant.name == "baseline") {
            continue;
        }
        printNote(row.variant.name.c_str(), row.variant.note);
    }

    printSubBanner("Extra: full mask only (same src/dst)");
    cv::Mat fullOut;
    double fullMs = 0.0;
    timeOnce([&]() { runFullMaskClone(bench, fullOut); }, fullMs);
    const CompareResult fullVsRect = compareImages(baselineOut, fullOut, 42.0, 6.0);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  full_mask (same src/dst): " << fullMs << " ms"
              << "   vs baseline " << baselineAvgMs << " ms"
              << "   speedup " << (baselineAvgMs / std::max(fullMs, 1e-6)) << "x\n";
    std::cout << "  quality vs rect mask:    PSNR=" << fullVsRect.psnr
              << " dB   maxDiff=" << fullVsRect.maxAbsDiff << "   "
              << (fullVsRect.pass ? "PASS" : "FAIL") << "\n";

    printBanner(identicalPass ? "SAME paths: PASS" : "SAME paths: FAIL");
    std::cout << "  SAME paths require maxDiff=0 (prealloc / pooled / threads)\n";
    std::cout << "  Images are written after this table (see Visual Export section).\n\n";

    return identicalPass ? 0 : 1;
}
