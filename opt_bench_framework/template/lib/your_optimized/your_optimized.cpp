#include "your_optimized.h"

#include <opencv2/imgproc.hpp>

namespace your_optimized {

void Context::gaussianBlur5x5(const cv::Mat& input, cv::Mat& output) {
    if (buf_.size() != input.size() || buf_.type() != input.type()) {
        buf_.create(input.size(), input.type());
    }
    cv::GaussianBlur(input, buf_, cv::Size(5, 5), 1.0);
    buf_.copyTo(output);
}

}  // namespace your_optimized
