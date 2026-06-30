#pragma once

#include "optimized_clone.h"

#include <string>
#include <vector>

struct VisualExportEntry {
    std::string fileStem;
    std::string title;
    cv::Mat image;
    double maxDiffVsBaseline = -1.0;
};

// Run all clone paths, write PNG/JPG into exportDir, return number of files written.
int runVisualExport(const BenchCase& bench, const std::string& exportDir);
