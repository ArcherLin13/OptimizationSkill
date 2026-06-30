#include "bench_case.h"

#include "image_io.h"
#include "seamless_roi.h"

#include <fstream>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kW = 729;
constexpr int kH = 126;
const cv::Rect kAppMaskRect(5, 5, 718, 115);

void applyAppMask(BenchCase& bench) {
    bench.mask = cv::Mat::zeros(kH, kW, CV_8UC1);
    bench.mask(kAppMaskRect).setTo(255);
    bench.center = cv::Point(364, 62);
}

void drawTextLines(cv::Mat& image, const std::vector<std::string>& lines, const cv::Scalar& color,
                   double scale, int thickness, int y0, int yStep) {
    int y = y0;
    for (const std::string& line : lines) {
        cv::putText(image, line, cv::Point(16, y), cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv::LINE_AA);
        y += yStep;
    }
}

}  // namespace

BenchCase makeVisualBenchCase() {
    BenchCase bench;
    bench.src = cv::Mat(kH, kW, CV_8UC3);
    bench.dst = cv::Mat(kH, kW, CV_8UC3);
    applyAppMask(bench);

    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < bench.src.cols; ++x) {
            const uchar dstB = static_cast<uchar>((x * 255) / std::max(1, bench.src.cols - 1));
            const uchar dstG = static_cast<uchar>(40 + (y * 160) / std::max(1, bench.src.rows - 1));
            bench.dst.at<cv::Vec3b>(y, x) = cv::Vec3b(dstB, dstG, 220);

            const uchar stripe = static_cast<uchar>(((x / 12) + (y / 8)) % 2 ? 220 : 40);
            bench.src.at<cv::Vec3b>(y, x) =
                cv::Vec3b(stripe, static_cast<uchar>(80 + (x * 120) / bench.src.cols),
                          static_cast<uchar>(200 - (y * 120) / bench.src.rows));
        }
    }

    return bench;
}

BenchCase makeTextBenchCase() {
    BenchCase bench;
    bench.src = cv::Mat(kH, kW, CV_8UC3, cv::Scalar(210, 230, 255));
    bench.dst = cv::Mat(kH, kW, CV_8UC3, cv::Scalar(90, 40, 25));
    applyAppMask(bench);

    drawTextLines(bench.dst,
                  {"BACKGROUND 729x126", "DST: ABCD EFGH IJKL MNOP QRST", "harmony seamless clone test"},
                  cv::Scalar(235, 235, 235), 0.55, 1, 34, 36);
    drawTextLines(bench.src,
                  {"SOURCE PATCH CLONE", "SRC: 0123456789 9876543210", "glyph edge sharpness check"},
                  cv::Scalar(20, 20, 160), 0.55, 2, 34, 36);

  // Fine serif-like stress: small digits along the bottom.
    for (int x = 20; x < kW - 20; x += 54) {
        cv::putText(bench.dst, "8gQy", cv::Point(x, 118), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.55,
                    cv::Scalar(200, 200, 120), 1, cv::LINE_AA);
        cv::putText(bench.src, "6pZj", cv::Point(x, 118), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.55,
                    cv::Scalar(180, 60, 20), 1, cv::LINE_AA);
    }

    return bench;
}

BenchCase makeBenchCaseByName(const std::string& name) {
    if (name == "visual" || name == "pattern") {
        return makeVisualBenchCase();
    }
    if (name == "text") {
        return makeTextBenchCase();
    }
    throw std::runtime_error("unknown --case: " + name + " (use visual or text)");
}

bool saveBenchCaseToDir(const BenchCase& bench, const std::string& dir) {
    return saveImageFile(dir + "/src.bmp", bench.src) && saveImageFile(dir + "/dst.bmp", bench.dst) &&
           saveImageFile(dir + "/mask.bmp", bench.mask);
}

bool loadBenchCaseFromDir(const std::string& dir, BenchCase& bench, std::string& error) {
    const std::string srcPath = dir + "/src.bmp";
    const std::string dstPath = dir + "/dst.bmp";
    const std::string maskPath = dir + "/mask.bmp";

    if (!loadImageFile(srcPath, bench.src, false)) {
        error = "missing or unreadable: " + srcPath;
        return false;
    }
    if (!loadImageFile(dstPath, bench.dst, false)) {
        error = "missing or unreadable: " + dstPath;
        return false;
    }
    if (!loadImageFile(maskPath, bench.mask, true)) {
        error = "missing or unreadable: " + maskPath;
        return false;
    }
    if (bench.src.size() != bench.dst.size() || bench.src.size() != bench.mask.size()) {
        error = "src/dst/mask size mismatch";
        return false;
    }

    bench.center = cv::Point(bench.dst.cols / 2, bench.dst.rows / 2);
    const std::string centerPath = dir + "/center.txt";
    std::ifstream centerFile(centerPath);
    if (centerFile) {
        int cx = bench.center.x;
        int cy = bench.center.y;
        char comma = ',';
        centerFile >> cx >> comma >> cy;
        if (centerFile.good() || centerFile.eof()) {
            bench.center = cv::Point(cx, cy);
        }
    }
    return true;
}

int solverBoundingRectPixels(const BenchCase& bench) {
    const cv::Mat mask = preprocessMaskLikeOpenCV(bench.mask);
    const cv::Rect roi = cv::boundingRect(mask);
    return roi.width * roi.height;
}
