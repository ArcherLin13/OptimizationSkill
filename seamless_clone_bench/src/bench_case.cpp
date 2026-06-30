#include "bench_case.h"

#include "image_io.h"
#include "seamless_roi.h"

#include <fstream>
#include <sstream>

BenchCase makeVisualBenchCase() {
    BenchCase bench;
    bench.src = cv::Mat(126, 729, CV_8UC3);
    bench.dst = cv::Mat(126, 729, CV_8UC3);
    bench.mask = cv::Mat::zeros(126, 729, CV_8UC1);
    bench.center = cv::Point(364, 62);

    for (int y = 0; y < bench.src.rows; ++y) {
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

    bench.mask(cv::Rect(5, 5, 718, 115)).setTo(255);
    return bench;
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
