#pragma once

#include <opencv2/core.hpp>

// OpenCV 4.9 NORMAL_CLONE reimplementation (DST/DFT). Verified maxDiff=0 vs cv::seamlessClone.

namespace seamless_clone_fft {

class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                       cv::Mat& output, int flags = 1);

private:
    struct Impl;
    Impl* impl_;
};

// One-shot (allocates internal Context per call). Prefer a long-lived Context in hot loops.
void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags = 1);

}  // namespace seamless_clone_fft
