#include "optimized_clone.h"
#include "metrics.h"
#include "timing.h"

#include <iomanip>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace {

BenchCase makeUserCase() {
  BenchCase bench;
  bench.src = cv::Mat(126, 729, CV_8UC3);
  bench.dst = cv::Mat(126, 729, CV_8UC3);
  bench.mask = cv::Mat::zeros(126, 729, CV_8UC1);
  bench.center = cv::Point(364, 62);

  for (int y = 0; y < bench.src.rows; ++y) {
    for (int x = 0; x < bench.src.cols; ++x) {
      bench.src.at<cv::Vec3b>(y, x) =
          cv::Vec3b(static_cast<uchar>((x * 3) % 256), static_cast<uchar>((y * 5) % 256),
                    static_cast<uchar>(((x + y) * 2) % 256));
      bench.dst.at<cv::Vec3b>(y, x) =
          cv::Vec3b(static_cast<uchar>((x + 40) % 256), static_cast<uchar>((y + 80) % 256),
                    static_cast<uchar>(((x + y + 120) * 2) % 256));
    }
  }

  cv::ellipse(bench.mask, bench.center, cv::Size(300, 50), 0, 0, 360, cv::Scalar(255), -1);
  return bench;
}

void printStats(const char* label, const TimingStats& stats) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << label << ": avg=" << stats.avgMs << "ms min=" << stats.minMs
            << "ms max=" << stats.maxMs << "ms\n";
}

void printCompare(const char* label, const CompareResult& cmp, const char* note) {
  std::cout << label << " correctness: psnr=" << cmp.psnr << "dB maxDiff=" << cmp.maxAbsDiff
            << " => " << (cmp.pass ? "PASS" : "FAIL") << " (" << note << ")\n";
}

using BaselineFn = void (*)(const BenchCase&, cv::Mat&);

void timeBaselineCall(const char* label, const BenchCase& bench, BaselineFn fn) {
  double ms = 0.0;
  cv::Mat out;
  timeOnce([&]() { fn(bench, out); }, ms);
  std::cout << std::fixed << std::setprecision(2) << label << ": " << ms << " ms\n";
}

}  // namespace

int runBenchmark() {
  std::cout << "OpenCV " << CV_VERSION << "\n";
  std::cout << "=== seamlessClone benchmark (NORMAL_CLONE) ===\n";

  const BenchCase bench = makeUserCase();
  std::cout << "case: src=" << bench.src.cols << "x" << bench.src.rows
            << " dst=" << bench.dst.cols << "x" << bench.dst.rows
            << " mask=" << bench.mask.cols << "x" << bench.mask.rows << " center=("
            << bench.center.x << "," << bench.center.y << ")"
            << " solver_px=" << (bench.src.cols * bench.src.rows) << "\n";

  std::cout << "--- cold / warmup (baseline only) ---\n";
  std::cout << "compare cold_first with your app's first seamlessClone call\n";
  timeBaselineCall("cold_first", bench, runBaselineClone);

  const auto baselineOnce = [&](cv::Mat& tmp) { runBaselineClone(bench, tmp); };

  std::cout << "--- timing (5 runs, NO warmup; 1 prior call = cold_first) ---\n";
  const TimingStats baselineNoWarm =
      measureMs(0, 5, [&]() {
        cv::Mat tmp;
        baselineOnce(tmp);
      });
  printStats("baseline", baselineNoWarm);

  double warmupTotal = 0.0;
  constexpr int kWarmupRuns = 2;
  for (int i = 0; i < kWarmupRuns; ++i) {
    double ms = 0.0;
    cv::Mat out;
    timeOnce([&]() { runBaselineClone(bench, out); }, ms);
    warmupTotal += ms;
    std::cout << std::fixed << std::setprecision(2) << "warmup[" << (i + 1) << "]: " << ms
              << " ms\n";
  }
  std::cout << std::fixed << std::setprecision(2) << "warmup_total: " << warmupTotal << " ms\n";

  std::cout << "--- timing (5 runs after 2 warmup above) ---\n";
  const TimingStats baselineWarm =
      measureMs(0, 5, [&]() {
        cv::Mat tmp;
        baselineOnce(tmp);
      });
  printStats("baseline", baselineWarm);

  std::cout << "--- correctness (after timing) ---\n";
  cv::Mat baselineOut;
  runBaselineClone(bench, baselineOut);

  cv::Mat pooledOut;
  PooledCloneContext ctx;
  ctx.srcBuf.create(bench.src.size(), bench.src.type());
  ctx.dstBuf.create(bench.dst.size(), bench.dst.type());
  ctx.maskBuf.create(bench.mask.size(), bench.mask.type());
  ctx.outBuf.create(bench.dst.size(), bench.dst.type());
  runPooledClone(ctx, bench, pooledOut);

  cv::Mat alignedOut;
  runAlignedClone(bench, alignedOut, 32);

  cv::Mat halfOut;
  runHalfResClone(bench, halfOut);

  const CompareResult pooledCmp = compareImages(baselineOut, pooledOut, 100.0, 0.0);
  const CompareResult alignedCmp = compareImages(baselineOut, alignedOut, 42.0, 6.0);
  const CompareResult halfCmp = compareImages(baselineOut, halfOut, 28.0, 25.0);

  printCompare("pooled_reuse", pooledCmp, "must be identical");
  printCompare("aligned_736x128", alignedCmp, "FFT-friendly pad, near-identical");
  printCompare("half_res", halfCmp, "approximate fast path");

  std::cout << "--- timing other paths (2 warmup + 5 runs) ---\n";
  const TimingStats pooledStats =
      measureMs(kWarmupRuns, 5, [&]() {
        cv::Mat tmp;
        runPooledClone(ctx, bench, tmp);
      });
  const TimingStats alignedStats =
      measureMs(kWarmupRuns, 5, [&]() {
        cv::Mat tmp;
        runAlignedClone(bench, tmp, 32);
      });
  const TimingStats halfStats =
      measureMs(kWarmupRuns, 5, [&]() {
        cv::Mat tmp;
        runHalfResClone(bench, tmp);
      });

  printStats("pooled_reuse", pooledStats);
  printStats("aligned_736x128", alignedStats);
  printStats("half_res", halfStats);

  const double speedupAligned = baselineWarm.avgMs / std::max(alignedStats.avgMs, 1e-6);
  const double speedupHalf = baselineWarm.avgMs / std::max(halfStats.avgMs, 1e-6);
  std::cout << "speedup aligned: " << speedupAligned << "x\n";
  std::cout << "speedup half_res: " << speedupHalf << "x\n";

  const bool allRequiredPass = pooledCmp.pass && alignedCmp.pass;
  std::cout << "=== overall: " << (allRequiredPass ? "PASS" : "FAIL") << " ===\n";
  return allRequiredPass ? 0 : 1;
}
