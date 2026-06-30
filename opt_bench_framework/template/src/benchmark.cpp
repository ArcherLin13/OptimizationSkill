#include "baseline.h"
#include "bench_case.h"
#include "log_format.h"
#include "optimized.h"
#include "timing.h"

#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace {

constexpr int kWarmup = 2;
constexpr int kRuns = 5;

enum class VariantKind { Reference, Identical, Fast, Approx };

const char* kindLabel(VariantKind k) {
    switch (k) {
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

struct Variant {
    std::string name;
    VariantKind kind;
    std::function<void(cv::Mat&)> run;
    double psnrThreshold;
    double maxDiffThreshold;
    const char* note;
};

struct Row {
    Variant v;
    TimingStats stats;
    CompareResult cmp;
    bool hasCmp = false;
};

}  // namespace

int runBenchmark(const BenchCase& bench) {
    printBanner("__BENCH_NAME__");

    printSubBanner("Environment");
    printKv("OpenCV", CV_VERSION);
    printKv("CPU threads", std::to_string(cv::getNumThreads()));
    printKv("case", bench.label);
    printKv("input", std::to_string(bench.input.cols) + "x" + std::to_string(bench.input.rows));

    cv::Mat baselineOut;
    runBaseline(bench.input, baselineOut);

    std::vector<Variant> variants;
    variants.push_back({"baseline", VariantKind::Reference,
                        [&](cv::Mat& out) { runBaseline(bench.input, out); }, 100.0, 0.0,
                        "naive alloc each call"});
    variants.push_back({"optimized", VariantKind::Identical,
                        [&](cv::Mat& out) { runOptimized(bench.input, out); }, 100.0, 0.0,
                        "lib/your_optimized Context reuse"});

    // TODO: add more variants (FAST/APRX) as you iterate

    std::vector<Row> rows;
    double baselineAvg = 0.0;

    printSubBanner("Legend");
    std::cout << "  kind  REF=baseline  SAME=must match  FAST/APRX=threshold in benchmark.cpp\n";
    std::cout << "  timing " << kWarmup << " warmup + " << kRuns << " runs\n";

    printResultTableHeader();

    for (const auto& v : variants) {
        Row row;
        row.v = v;
        row.stats = measureMs(kWarmup, kRuns, [&]() {
            cv::Mat out;
            v.run(out);
        });

        if (v.kind == VariantKind::Reference) {
            baselineAvg = row.stats.avgMs;
        }

        if (v.kind != VariantKind::Reference) {
            cv::Mat out;
            v.run(out);
            row.cmp = compareImages(baselineOut, out, v.psnrThreshold, v.maxDiffThreshold);
            row.hasCmp = true;
        }

        const double speedup = (baselineAvg > 0.0) ? baselineAvg / row.stats.avgMs : 0.0;
        printResultRow(v.name.c_str(), kindLabel(v.kind), row.stats, speedup,
                       row.hasCmp ? &row.cmp : nullptr, row.hasCmp);
        rows.push_back(row);
    }

    printSubBanner("Path notes");
    for (const auto& row : rows) {
        std::cout << "      -> " << row.v.name << ": " << row.v.note << "\n";
    }

    bool allSamePass = true;
    for (const auto& row : rows) {
        if (row.v.kind == VariantKind::Identical && row.hasCmp && !row.cmp.pass) {
            allSamePass = false;
        }
    }
    printSubBanner(allSamePass ? "SAME paths: PASS" : "SAME paths: FAIL");
    return allSamePass ? 0 : 1;
}
