#pragma once

#include <opencv2/core.hpp>

// Phase A: OpenCV 4.9 NORMAL_CLONE reimplementation (DST Poisson solver).
// Goal: maxDiff=0 vs cv::seamlessClone on device before any speed tuning.

namespace seamless_clone_fft {

void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags = 1);

}  // namespace seamless_clone_fft
