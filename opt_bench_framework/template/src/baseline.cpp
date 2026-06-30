#include "baseline.h"

#include <opencv2/imgproc.hpp>

// Example baseline: allocate output every call (like naive main-project code).
void runBaseline(const cv::Mat& input, cv::Mat& output) {
    cv::Mat tmp;
    cv::GaussianBlur(input, tmp, cv::Size(5, 5), 1.0);
    tmp.copyTo(output);
}
