#include "optimized.h"

#include "your_optimized.h"

void runOptimized(const cv::Mat& input, cv::Mat& output) {
    static your_optimized::Context ctx;
    ctx.gaussianBlur5x5(input, output);
}
