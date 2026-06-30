#include "visual_export.h"

#include "bench_case.h"
#include "image_io.h"
#include "log_format.h"
#include "metrics.h"
#include "poisson_jacobi.h"

#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace {

constexpr int kJacobiIterations = 400;
constexpr int kTileScale = 3;

std::string joinPath(const std::string& dir, const std::string& name) {
    return (fs::path(dir) / name).string();
}

bool saveImage(const std::string& path, const cv::Mat& image) {
    return saveImageFile(path, image);
}

cv::Mat colorizeMask(const cv::Mat& mask) {
    cv::Mat color(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    color.setTo(cv::Scalar(0, 220, 0), mask);
    return color;
}

cv::Mat amplifyDiff(const cv::Mat& baseline, const cv::Mat& other, double scale = 8.0) {
    cv::Mat diff;
    cv::absdiff(baseline, other, diff);
    cv::Mat vis;
    diff.convertTo(vis, CV_8UC3, scale);
    return vis;
}

cv::Mat makeTile(const cv::Mat& image, const std::string& title, int maxDiff) {
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image;
    }

    cv::Mat scaled;
    cv::resize(bgr, scaled, cv::Size(), kTileScale, kTileScale, cv::INTER_NEAREST);

    const int barH = 36;
    cv::Mat tile(scaled.rows + barH, scaled.cols, CV_8UC3, cv::Scalar(24, 24, 24));
    scaled.copyTo(tile(cv::Rect(0, barH, scaled.cols, scaled.rows)));

    std::ostringstream label;
    label << title;
    if (maxDiff >= 0) {
        label << "  maxDiff=" << maxDiff;
    }
    cv::putText(tile, label.str(), cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    return tile;
}

cv::Mat stitchRows(const std::vector<cv::Mat>& tiles, int columns) {
    if (tiles.empty()) {
        return {};
    }

    const int cols = std::max(1, columns);
    const int rows = static_cast<int>((tiles.size() + cols - 1) / cols);
    const int tileW = tiles[0].cols;
    const int tileH = tiles[0].rows;

    cv::Mat canvas(rows * tileH, cols * tileW, CV_8UC3, cv::Scalar(16, 16, 16));
    for (size_t i = 0; i < tiles.size(); ++i) {
        const int r = static_cast<int>(i / cols);
        const int c = static_cast<int>(i % cols);
        cv::Mat resized = tiles[i];
        if (resized.cols != tileW || resized.rows != tileH) {
            cv::resize(tiles[i], resized, cv::Size(tileW, tileH));
        }
        resized.copyTo(canvas(cv::Rect(c * tileW, r * tileH, tileW, tileH)));
    }
    return canvas;
}

void addVariant(std::vector<VisualExportEntry>& entries, const std::string& stem, const std::string& title,
                const cv::Mat& image, const cv::Mat& baseline) {
    VisualExportEntry entry;
    entry.fileStem = stem;
    entry.title = title;
    entry.image = image;
    if (!baseline.empty() && !image.empty()) {
        cv::Mat diff;
        cv::absdiff(baseline, image, diff);
        double maxDiff = 0.0;
        cv::minMaxLoc(diff, nullptr, &maxDiff);
        entry.maxDiffVsBaseline = maxDiff;
    }
    entries.push_back(entry);
}

}  // namespace

int runVisualExport(const BenchCase& bench, const std::string& exportDir) {
    fs::create_directories(exportDir);

    PooledCloneContext pool;
    pool.srcBuf.create(bench.src.size(), bench.src.type());
    pool.dstBuf.create(bench.dst.size(), bench.dst.type());
    pool.maskBuf.create(bench.mask.size(), bench.mask.type());
    pool.outBuf.create(bench.dst.size(), bench.dst.type());

    cv::Mat preallocOut;
    preallocOut.create(bench.dst.size(), bench.dst.type());

    cv::Mat baseline;
    runBaselineClone(bench, baseline);

    std::vector<VisualExportEntry> entries;
    addVariant(entries, "00_src", "src", bench.src, cv::Mat());
    addVariant(entries, "01_dst", "dst", bench.dst, cv::Mat());
    addVariant(entries, "02_mask", "mask", colorizeMask(bench.mask), cv::Mat());
    addVariant(entries, "03_baseline", "baseline", baseline, cv::Mat());

    cv::Mat out;
    runPreallocOutClone(bench, preallocOut);
    addVariant(entries, "04_prealloc_out", "prealloc_out", preallocOut, baseline);

    runPooledClone(pool, bench, out);
    addVariant(entries, "05_pooled_reuse", "pooled_reuse", out, baseline);

    runFullMaskClone(bench, out);
    addVariant(entries, "06_full_mask", "full_mask", out, baseline);

    runFullMaskBorderPasteClone(bench, out, 5);
    addVariant(entries, "07_full_mask_border_paste5", "full_mask_border_paste5", out, baseline);

    runAlignedClone(bench, out, 32);
    addVariant(entries, "08_aligned_736x128", "aligned_736x128", out, baseline);

    runHalfResClone(bench, out);
    addVariant(entries, "09_half_res", "half_res", out, baseline);

    runJacobiPoissonClone(bench, out, kJacobiIterations, false);
    addVariant(entries, "10_jacobi_cpu", "jacobi_cpu", out, baseline);

    if (isOpenCLPoissonAvailable()) {
        runJacobiPoissonClone(bench, out, kJacobiIterations, true);
        addVariant(entries, "11_jacobi_opencl", "jacobi_opencl", out, baseline);
    }

    int written = 0;
    printBanner("Visual Export");
    printKv("export dir", exportDir);
    printSubBanner("Output files");

    for (const VisualExportEntry& entry : entries) {
        const std::string path = joinPath(exportDir, entry.fileStem + ".bmp");
        if (!saveImage(path, entry.image)) {
            std::cout << "  [FAIL] " << path << "\n";
            continue;
        }
        ++written;
        std::cout << "  " << entry.fileStem << ".bmp";
        if (entry.maxDiffVsBaseline >= 0.0) {
            std::cout << "   maxDiff=" << static_cast<int>(entry.maxDiffVsBaseline);
        }
        std::cout << "\n";
    }

    std::vector<cv::Mat> resultTiles;
    std::vector<cv::Mat> diffTiles;
    for (const VisualExportEntry& entry : entries) {
        const bool isInput = entry.fileStem == "00_src" || entry.fileStem == "01_dst" ||
                             entry.fileStem == "02_mask" || entry.fileStem == "03_baseline";
        const int maxDiff = entry.maxDiffVsBaseline >= 0.0 ? static_cast<int>(entry.maxDiffVsBaseline) : -1;
        if (isInput) {
            resultTiles.push_back(makeTile(entry.image, entry.title, -1));
            continue;
        }
        resultTiles.push_back(makeTile(entry.image, entry.title, maxDiff));
        diffTiles.push_back(makeTile(amplifyDiff(baseline, entry.image), "diff " + entry.title, maxDiff));
    }

    const cv::Mat gridResults = stitchRows(resultTiles, 3);
    const cv::Mat gridDiffs = stitchRows(diffTiles, 3);
    const std::string gridResultsPath = joinPath(exportDir, "grid_results.bmp");
    const std::string gridDiffsPath = joinPath(exportDir, "grid_diff_x8.bmp");

    if (saveImage(gridResultsPath, gridResults)) {
        ++written;
        std::cout << "  grid_results.bmp   (all outputs, 3x zoom)\n";
    }
    if (saveImage(gridDiffsPath, gridDiffs)) {
        ++written;
        std::cout << "  grid_diff_x8.bmp   (absdiff vs baseline, x8)\n";
    }

    printSubBanner("How to view");
    std::cout << "  on device:  " << exportDir << "\n";
    std::cout << "  pull to PC: .\\scripts\\pull_results.ps1\n";
    std::cout << "  key files:  grid_results.bmp / grid_diff_x8.bmp\n\n";

    return written;
}
