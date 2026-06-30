#include "seamless_clone_jacobi.h"

#include "seamless_roi.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void computeGradientX(const cv::Mat& img, cv::Mat& gx) {
    const cv::Mat kernel = (cv::Mat_<char>(1, 3) << 0, -1, 1);
    cv::filter2D(img, gx, CV_32F, kernel);
    if (img.channels() == 1) {
        cv::cvtColor(gx, gx, cv::COLOR_GRAY2BGR);
    }
}

void computeGradientY(const cv::Mat& img, cv::Mat& gy) {
    const cv::Mat kernel = (cv::Mat_<char>(3, 1) << 0, -1, 1);
    cv::filter2D(img, gy, CV_32F, kernel);
    if (img.channels() == 1) {
        cv::cvtColor(gy, gy, cv::COLOR_GRAY2BGR);
    }
}

void computeLaplacianX(const cv::Mat& img, cv::Mat& lx) {
    const cv::Mat kernel = (cv::Mat_<char>(1, 3) << -1, 1, 0);
    cv::filter2D(img, lx, CV_32F, kernel);
}

void computeLaplacianY(const cv::Mat& img, cv::Mat& ly) {
    const cv::Mat kernel = (cv::Mat_<char>(3, 1) << -1, 1, 0);
    cv::filter2D(img, ly, CV_32F, kernel);
}

void buildNormalCloneLaplacian(const cv::Mat& destination, const cv::Mat& patch, cv::Mat binaryMask,
                               std::vector<cv::Mat>& lapPerChannel) {
    cv::Mat destGx;
    cv::Mat destGy;
    cv::Mat patchGx;
    cv::Mat patchGy;
    computeGradientX(destination, destGx);
    computeGradientY(destination, destGy);
    computeGradientX(patch, patchGx);
    computeGradientY(patch, patchGy);

    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::erode(binaryMask, binaryMask, kernel, cv::Point(-1, -1), 3);

    cv::Mat maskFloat;
    binaryMask.convertTo(maskFloat, CV_32FC1, 1.0 / 255.0);

    cv::Mat invMask;
    cv::bitwise_not(binaryMask, invMask);
    cv::Mat invMaskFloat;
    invMask.convertTo(invMaskFloat, CV_32FC1, 1.0 / 255.0);

    std::vector<cv::Mat> patchGxCh;
    std::vector<cv::Mat> patchGyCh;
    std::vector<cv::Mat> destGxCh;
    std::vector<cv::Mat> destGyCh;
    cv::split(patchGx, patchGxCh);
    cv::split(patchGy, patchGyCh);
    cv::split(destGx, destGxCh);
    cv::split(destGy, destGyCh);

    lapPerChannel.resize(3);
    for (int c = 0; c < 3; ++c) {
        cv::multiply(patchGxCh[c], maskFloat, patchGxCh[c]);
        cv::multiply(patchGyCh[c], maskFloat, patchGyCh[c]);
        cv::multiply(destGxCh[c], invMaskFloat, destGxCh[c]);
        cv::multiply(destGyCh[c], invMaskFloat, destGyCh[c]);

        const cv::Mat gx = destGxCh[c] + patchGxCh[c];
        const cv::Mat gy = destGyCh[c] + patchGyCh[c];
        cv::Mat lx;
        cv::Mat ly;
        computeLaplacianX(gx, lx);
        computeLaplacianY(gy, ly);
        lapPerChannel[c] = lx + ly;
    }
}

void jacobiChannelCpu(const cv::Mat& lap, const cv::Mat& imgU8, cv::Mat& outU8, int iterations) {
    const int w = imgU8.cols;
    const int h = imgU8.rows;

    cv::Mat u;
    imgU8.convertTo(u, CV_32F);

    const cv::Mat f = lap(cv::Rect(1, 1, w - 2, h - 2));
    cv::Mat a = u.clone();
    cv::Mat b = u.clone();

    for (int iter = 0; iter < iterations; ++iter) {
        const cv::Mat& src = (iter % 2 == 0) ? a : b;
        cv::Mat& dst = (iter % 2 == 0) ? b : a;

        cv::parallel_for_(cv::Range(1, h - 1), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const float* srcAbove = src.ptr<float>(y - 1);
                const float* srcRow = src.ptr<float>(y);
                const float* srcBelow = src.ptr<float>(y + 1);
                const float* fRow = f.ptr<float>(y - 1);
                float* dstRow = dst.ptr<float>(y);

                for (int x = 1; x < w - 1; ++x) {
                    dstRow[x] =
                        0.25f * (srcAbove[x] + srcBelow[x] + srcRow[x - 1] + srcRow[x + 1] - fRow[x - 1]);
                }
            }
        });
    }

    const cv::Mat& result = (iterations % 2 == 0) ? a : b;
    outU8.create(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        const float* srcRow = result.ptr<float>(y);
        uchar* dstRow = outU8.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const float value = srcRow[x];
            dstRow[x] = static_cast<uchar>(std::min(255.f, std::max(0.f, value)));
        }
    }
}

bool runJacobiNormalClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                          cv::Mat& output, int iterations) {
    SeamlessRoi roi;
    if (!extractSeamlessRoi(src, dst, mask, center, roi)) {
        return false;
    }

    cv::Mat binaryMask = roi.maskROI.clone();
    std::vector<cv::Mat> lapPerChannel;
    buildNormalCloneLaplacian(roi.dstROI, roi.srcROI, binaryMask, lapPerChannel);

    std::vector<cv::Mat> channels;
    cv::split(roi.dstROI, channels);
    for (int c = 0; c < 3; ++c) {
        cv::Mat solved;
        jacobiChannelCpu(lapPerChannel[c], channels[c], solved, iterations);
        channels[c] = solved;
    }

    cv::merge(channels, roi.dstROI);
    dst.copyTo(output);
    roi.dstROI.copyTo(output(roi.roi_d));
    return true;
}

}  // namespace

namespace seamless_clone_jacobi {

void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags, int iterations) {
    if (flags != 1) {  // cv::NORMAL_CLONE
        throw std::invalid_argument("seamless_clone_jacobi::seamlessClone only supports NORMAL_CLONE");
    }
    if (src.empty() || dst.empty() || mask.empty()) {
        throw std::invalid_argument("seamless_clone_jacobi::seamlessClone: empty input");
    }
    if (src.size() != dst.size() || src.size() != mask.size()) {
        throw std::invalid_argument("seamless_clone_jacobi::seamlessClone: src/dst/mask size mismatch");
    }
    if (src.type() != CV_8UC3 || dst.type() != CV_8UC3) {
        throw std::invalid_argument("seamless_clone_jacobi::seamlessClone: src/dst must be CV_8UC3");
    }
    if (!runJacobiNormalClone(src, dst, mask, center, output, iterations)) {
        throw std::runtime_error("seamless_clone_jacobi::seamlessClone: invalid mask or ROI");
    }
}

}  // namespace seamless_clone_jacobi
