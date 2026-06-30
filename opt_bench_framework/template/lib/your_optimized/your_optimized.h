#pragma once

#include <opencv2/core.hpp>

namespace your_optimized {

// Reuse buffers across frames (your real optimization goes here).
class Context {
public:
    void gaussianBlur5x5(const cv::Mat& input, cv::Mat& output);

private:
    cv::Mat buf_;
};

}  // namespace your_optimized
