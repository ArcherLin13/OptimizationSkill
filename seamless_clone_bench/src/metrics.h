#pragma once

#include <cmath>
#include <opencv2/core.hpp>

struct CompareResult {
    double psnr = 0.0;
    double maxAbsDiff = 0.0;
    bool pass = false;
};

inline CompareResult compareImages(const cv::Mat& a, const cv::Mat& b, double psnrThreshold,
                                   double maxDiffThreshold) {
    CV_Assert(a.size() == b.size());
    CV_Assert(a.type() == b.type());

    cv::Mat diff;
    cv::absdiff(a, b, diff);

    double maxDiff = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxDiff);

    cv::Mat a32;
    cv::Mat b32;
    if (a.depth() == CV_32F) {
        a32 = a;
    } else {
        a.convertTo(a32, CV_32F);
    }
    if (b.depth() == CV_32F) {
        b32 = b;
    } else {
        b.convertTo(b32, CV_32F);
    }

    const double psnr = cv::PSNR(a32, b32);

    CompareResult result;
    result.psnr = psnr;
    result.maxAbsDiff = maxDiff;
    result.pass = (psnr >= psnrThreshold) && (maxDiff <= maxDiffThreshold);
    return result;
}
