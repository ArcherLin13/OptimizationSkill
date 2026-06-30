#pragma once

#include <opencv2/core.hpp>

struct CompareResult {
    double psnr = 0.0;
    double maxAbsDiff = 0.0;
    bool pass = false;
};

inline CompareResult compareImages(const cv::Mat& a, const cv::Mat& b, double psnrThreshold,
                                   double maxDiffThreshold) {
    CV_Assert(a.size() == b.size() && a.type() == b.type());
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    double maxDiff = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxDiff);
    cv::Mat a32, b32;
    a.convertTo(a32, CV_32F);
    b.convertTo(b32, CV_32F);
    CompareResult r;
    r.psnr = cv::PSNR(a32, b32);
    r.maxAbsDiff = maxDiff;
    r.pass = (r.psnr >= psnrThreshold) && (maxDiff <= maxDiffThreshold);
    return r;
}
