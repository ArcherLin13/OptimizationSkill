#pragma once

#include "optimized_clone.h"

#include <opencv2/imgproc.hpp>

struct SeamlessRoi {
    cv::Mat srcROI;
    cv::Mat dstROI;
    cv::Mat maskROI;
    cv::Rect roi_s;
    cv::Rect roi_d;
};

inline cv::Mat preprocessMaskLikeOpenCV(cv::Mat mask) {
    cv::Mat gray;
    if (mask.channels() > 1) {
        cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
    } else if (mask.empty()) {
        gray = cv::Mat(mask.rows, mask.cols, CV_8UC1, cv::Scalar(255));
    } else {
        gray = mask.clone();
    }

    const cv::Mat maskInner = gray(cv::Rect(1, 1, gray.cols - 2, gray.rows - 2));
    cv::Mat processed;
    cv::copyMakeBorder(maskInner, processed, 1, 1, 1, 1, cv::BORDER_ISOLATED | cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    return processed;
}

inline bool extractSeamlessRoi(const BenchCase& bench, SeamlessRoi& out) {
    const cv::Mat mask = preprocessMaskLikeOpenCV(bench.mask);
    out.roi_s = cv::boundingRect(mask);
    if (out.roi_s.empty()) {
        return false;
    }

    out.roi_d = cv::Rect(bench.center.x - out.roi_s.width / 2, bench.center.y - out.roi_s.height / 2,
                         out.roi_s.width, out.roi_s.height);

    out.dstROI = bench.dst(out.roi_d).clone();
    out.srcROI = cv::Mat::zeros(out.roi_s.size(), bench.src.type());
    bench.src(out.roi_s).copyTo(out.srcROI, mask(out.roi_s));
    out.maskROI = mask(out.roi_s);
    return true;
}

inline void writeSeamlessRoi(cv::Mat& blend, const cv::Mat& dest, const SeamlessRoi& roi) {
    dest.copyTo(blend);
    roi.dstROI.copyTo(blend(roi.roi_d));
}
