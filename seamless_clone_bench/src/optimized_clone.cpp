#include "optimized_clone.h"

#include "seamless_roi.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

cv::Size alignedSize(const cv::Size& size, int align) {
    const int w = ((size.width + align - 1) / align) * align;
    const int h = ((size.height + align - 1) / align) * align;
    return {w, h};
}

static cv::Mat padMat(const cv::Mat& src, const cv::Size& target) {
    cv::Mat padded(target, src.type(), cv::Scalar::all(0));
    const int x = (target.width - src.cols) / 2;
    const int y = (target.height - src.rows) / 2;
    const cv::Rect roi(x, y, src.cols, src.rows);
    src.copyTo(padded(roi));
    return padded;
}

static cv::Mat padMask(const cv::Mat& mask, const cv::Size& target) {
    cv::Mat padded(target, CV_8UC1, cv::Scalar(0));
    const int x = (target.width - mask.cols) / 2;
    const int y = (target.height - mask.rows) / 2;
    const cv::Rect roi(x, y, mask.cols, mask.rows);
    mask.copyTo(padded(roi));
    return padded;
}

void runBaselineClone(const BenchCase& bench, cv::Mat& output) {
    cv::seamlessClone(bench.src, bench.dst, bench.mask, bench.center, output, cv::NORMAL_CLONE);
}

void runBaselineCloneThreads(const BenchCase& bench, cv::Mat& output, int threads) {
    const int prev = cv::getNumThreads();
    cv::setNumThreads(threads);
    cv::seamlessClone(bench.src, bench.dst, bench.mask, bench.center, output, cv::NORMAL_CLONE);
    cv::setNumThreads(prev);
}

void runPreallocOutClone(const BenchCase& bench, cv::Mat& output) {
    output.create(bench.dst.size(), bench.dst.type());
    cv::seamlessClone(bench.src, bench.dst, bench.mask, bench.center, output, cv::NORMAL_CLONE);
}

void runPooledClone(PooledCloneContext& ctx, const BenchCase& bench, cv::Mat& output) {
    bench.src.copyTo(ctx.srcBuf);
    bench.dst.copyTo(ctx.dstBuf);
    bench.mask.copyTo(ctx.maskBuf);
    cv::seamlessClone(ctx.srcBuf, ctx.dstBuf, ctx.maskBuf, bench.center, ctx.outBuf, cv::NORMAL_CLONE);
    ctx.outBuf.copyTo(output);
}

void runFullMaskClone(const BenchCase& bench, cv::Mat& output) {
    const cv::Mat fullMask(bench.mask.size(), CV_8UC1, cv::Scalar(255));
    cv::seamlessClone(bench.src, bench.dst, fullMask, bench.center, output, cv::NORMAL_CLONE);
}

void runFullMaskBorderPasteClone(const BenchCase& bench, cv::Mat& output, int borderPx) {
    runFullMaskClone(bench, output);

    cv::Mat keepDst;
    if (borderPx > 0) {
        keepDst = cv::Mat::zeros(bench.mask.size(), CV_8UC1);
        const cv::Rect inner(borderPx, borderPx, bench.mask.cols - 2 * borderPx,
                             bench.mask.rows - 2 * borderPx);
        keepDst(inner).setTo(255);
        cv::bitwise_not(keepDst, keepDst);
    } else {
        cv::bitwise_not(bench.mask, keepDst);
    }
    bench.dst.copyTo(output, keepDst);
}

void runAlignedClone(const BenchCase& bench, cv::Mat& output, int align) {
    const cv::Size target = alignedSize(bench.src.size(), align);
    if (target == bench.src.size()) {
        runBaselineClone(bench, output);
        return;
    }

    const cv::Mat srcPad = padMat(bench.src, target);
    const cv::Mat dstPad = padMat(bench.dst, target);
    const cv::Mat maskPad = padMask(bench.mask, target);
    const cv::Point centerPad(target.width / 2, target.height / 2);

    cv::Mat outPad;
    cv::seamlessClone(srcPad, dstPad, maskPad, centerPad, outPad, cv::NORMAL_CLONE);

    const int x = (target.width - bench.src.cols) / 2;
    const int y = (target.height - bench.src.rows) / 2;
    outPad(cv::Rect(x, y, bench.src.cols, bench.src.rows)).copyTo(output);
}

void runHalfResClone(const BenchCase& bench, cv::Mat& output) {
    cv::Mat srcSmall;
    cv::Mat dstSmall;
    cv::Mat maskSmall;
    cv::resize(bench.src, srcSmall, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(bench.dst, dstSmall, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(bench.mask, maskSmall, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    cv::threshold(maskSmall, maskSmall, 127, 255, cv::THRESH_BINARY);

    const cv::Point centerSmall(bench.center.x / 2, bench.center.y / 2);
    cv::Mat outSmall;
    cv::seamlessClone(srcSmall, dstSmall, maskSmall, centerSmall, outSmall, cv::NORMAL_CLONE);
    cv::resize(outSmall, output, bench.dst.size(), 0, 0, cv::INTER_LINEAR);
}
