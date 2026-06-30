#pragma once

#include <opencv2/core.hpp>

// CPU Jacobi Poisson solver — drop-in replacement for cv::seamlessClone(..., NORMAL_CLONE).
//
// Copy into your app (needs OpenCV core + imgproc only, not photo):
//   seamless_clone_jacobi.h
//   seamless_roi.h
//   poisson_jacobi.cpp
//
// Usage:
//   #include "seamless_clone_jacobi.h"
//   seamless_clone_jacobi::seamlessClone(src, dst, mask, center, output);
//
// Or swap the call site:
//   // cv::seamlessClone(src, dst, mask, center, output, cv::NORMAL_CLONE);
//   seamless_clone_jacobi::seamlessClone(src, dst, mask, center, output);

namespace seamless_clone_jacobi {

constexpr int kDefaultIterations = 400;

// Same inputs as cv::seamlessClone. Only cv::NORMAL_CLONE (1) is supported.
// src/dst: CV_8UC3, same size. mask: CV_8U (1 or 3 channels). center: anchor in dst.
// output: resized/reallocated to match dst if needed.
void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags = 1, int iterations = kDefaultIterations);

}  // namespace seamless_clone_jacobi
