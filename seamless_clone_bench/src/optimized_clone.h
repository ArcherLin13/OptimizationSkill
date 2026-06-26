#pragma once

#include <opencv2/core.hpp>

struct BenchCase {
    cv::Mat src;
    cv::Mat dst;
    cv::Mat mask;
    cv::Point center;
};

struct PooledCloneContext {
    cv::Mat srcBuf;
    cv::Mat dstBuf;
    cv::Mat maskBuf;
    cv::Mat outBuf;
};

void runBaselineClone(const BenchCase& bench, cv::Mat& output);

void runPooledClone(PooledCloneContext& ctx, const BenchCase& bench, cv::Mat& output);

void runAlignedClone(const BenchCase& bench, cv::Mat& output, int align = 32);

void runHalfResClone(const BenchCase& bench, cv::Mat& output);

cv::Size alignedSize(const cv::Size& size, int align);
