#pragma once

#include <opencv2/core.hpp>

// CPU-only drop-in for cv::seamlessClone(..., NORMAL_CLONE).
//
// Copy this entire folder into your project:
//   lib/seamless_clone_jacobi/
//     seamless_clone_jacobi.h
//     seamless_roi.h
//     seamless_clone_jacobi.cpp
//
// Add seamless_clone_jacobi.cpp to your build. Link OpenCV core + imgproc.
//
//   #include "seamless_clone_jacobi.h"
//   seamless_clone_jacobi::seamlessClone(src, dst, mask, center, output);

namespace seamless_clone_jacobi {

constexpr int kDefaultIterations = 400;

void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags = 1, int iterations = kDefaultIterations);

}  // namespace seamless_clone_jacobi
