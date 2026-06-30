#pragma once

#include <opencv2/imgproc.hpp>

bool saveImageFile(const std::string& path, const cv::Mat& image);
bool loadImageFile(const std::string& path, cv::Mat& image, bool grayscale = false);
