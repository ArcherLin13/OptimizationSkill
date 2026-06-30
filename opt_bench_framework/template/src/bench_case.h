#pragma once

#include <opencv2/core.hpp>
#include <string>

struct BenchCase {
    cv::Mat input;
    std::string label;
};

BenchCase makeDefaultCase();
BenchCase loadCaseFromDir(const std::string& dir);
