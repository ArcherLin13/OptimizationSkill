#pragma once

#include <opencv2/core.hpp>

// OpenCV 4.9 NORMAL_CLONE reimplementation (DST/DFT). Phase A: maxDiff=0 verified.
// Phase B: reuse buffers + parallel channel solve (still bit-identical).

namespace seamless_clone_fft {

class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                       cv::Mat& output, int flags = 1);

private:
    struct Impl;
    Impl* impl_;
};

// One-shot (allocates internal Context per call). Prefer a long-lived Context in hot loops.
void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags = 1);

// Skip cv::seamlessClone when src/dst/mask unchanged. Returns true if output was reused.
struct SkipState {
    uint64_t fingerprint = 0;
    cv::Mat cachedOutput;
};

bool seamlessCloneSkipUnchanged(SkipState& state, const cv::Mat& src, const cv::Mat& dst,
                                const cv::Mat& mask, cv::Point center, cv::Mat& output, int flags = 1,
                                Context* ctx = nullptr);

uint64_t fingerprintInputs(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center);

}  // namespace seamless_clone_fft
