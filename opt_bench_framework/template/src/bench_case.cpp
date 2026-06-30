#include "bench_case.h"

#include <opencv2/core.hpp>

BenchCase makeDefaultCase() {
    BenchCase c;
    c.label = "synthetic_720x128";
    c.input.create(128, 720, CV_8UC3);
    for (int y = 0; y < c.input.rows; ++y) {
        for (int x = 0; x < c.input.cols; ++x) {
            c.input.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>((x + y) % 256),
                static_cast<uchar>((x * 2) % 256),
                static_cast<uchar>((y * 3) % 256));
        }
    }
    return c;
}

BenchCase loadCaseFromDir(const std::string& dir) {
    // TODO: cv::imread(dir + "/input.png") when you add imgcodecs
    (void)dir;
    return makeDefaultCase();
}
