// OpenCV 4.9 NORMAL_CLONE + buffer reuse + 3-channel parallel (cv::dft merge path).
#include "seamless_clone_fft.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SC_FFT_NEON 1
#endif

namespace {

constexpr int kNormalClone = 1;

#ifdef SC_FFT_NEON

static void neonFlipNegateRow(const float* src, float* dstExt, int cols) {
    int i = 0;
    for (; i + 4 <= cols; i += 4) {
        const int base = cols - 4 - i;
        float32x4_t v = vld1q_f32(src + base);
        v = vrev64q_f32(v);
        v = vcombine_f32(vget_high_f32(v), vget_low_f32(v));
        vst1q_f32(dstExt + cols + 2 + i, vnegq_f32(v));
    }
    for (; i < cols; ++i) {
        dstExt[cols + 2 + i] = -src[cols - 1 - i];
    }
}

static void neonDivideEigenRow(float* row, int w, float fy, const float* fx) {
    int i = 0;
    const float32x4_t four = vdupq_n_f32(4.0f);
    const float32x4_t fyv = vdupq_n_f32(fy);
    for (; i + 4 <= w; i += 4) {
        const float32x4_t fxv = vld1q_f32(fx + i);
        const float32x4_t denom = vsubq_f32(vaddq_f32(fxv, fyv), four);
        float32x4_t v = vld1q_f32(row + i);
        v = vdivq_f32(v, denom);
        vst1q_f32(row + i, v);
    }
    for (; i < w; ++i) {
        row[i] /= fx[i] + fy - 4.0f;
    }
}

static void neonClampU8Row(uchar* dst, const float* interp, int wInner) {
    int x = 1;
    const int end = wInner - 1;
    for (; x + 4 <= end; x += 4) {
        float32x4_t v = vld1q_f32(interp + x - 1);
        v = vmaxq_f32(v, vdupq_n_f32(0.f));
        v = vminq_f32(v, vdupq_n_f32(255.f));
        uint32x4_t u32 = vcvtq_u32_f32(v);
        dst[x] = static_cast<uchar>(vgetq_lane_u32(u32, 0));
        dst[x + 1] = static_cast<uchar>(vgetq_lane_u32(u32, 1));
        dst[x + 2] = static_cast<uchar>(vgetq_lane_u32(u32, 2));
        dst[x + 3] = static_cast<uchar>(vgetq_lane_u32(u32, 3));
    }
    for (; x < end; ++x) {
        const float value = interp[x - 1];
        if (value < 0.f) {
            dst[x] = 0;
        } else if (value > 255.f) {
            dst[x] = 255;
        } else {
            dst[x] = static_cast<uchar>(value);
        }
    }
}

#else

static void neonFlipNegateRow(const float* src, float* dstExt, int cols) {
    for (int i = 0; i < cols; ++i) {
        dstExt[cols + 2 + i] = -src[cols - 1 - i];
    }
}

static void neonDivideEigenRow(float* row, int w, float fy, const float* fx) {
    for (int i = 0; i < w; ++i) {
        row[i] /= fx[i] + fy - 4.0f;
    }
}

static void neonClampU8Row(uchar* dst, const float* interp, int wInner) {
    for (int i = 1; i < wInner - 1; ++i) {
        const float value = interp[i - 1];
        if (value < 0.f) {
            dst[i] = 0;
        } else if (value > 255.f) {
            dst[i] = 255;
        } else {
            dst[i] = static_cast<uchar>(value);
        }
    }
}

#endif

struct EigenFilters {
    std::vector<float> x;
    std::vector<float> y;
};

EigenFilters makeFilters(int w, int h) {
    EigenFilters f;
    f.x.resize(static_cast<size_t>(w - 2));
    double scale = CV_PI / (w - 1);
    for (int i = 0; i < w - 2; ++i) {
        f.x[static_cast<size_t>(i)] = static_cast<float>(2.0 * std::cos(scale * (i + 1)));
    }
    f.y.resize(static_cast<size_t>(h - 2));
    scale = CV_PI / (h - 1);
    for (int j = 0; j < h - 2; ++j) {
        f.y[static_cast<size_t>(j)] = static_cast<float>(2.0 * std::cos(scale * (j + 1)));
    }
    return f;
}

struct DstScratch {
    cv::Mat tempA;
    cv::Mat tempB;
    cv::Mat complexA;
    cv::Mat complexB;
    cv::Mat planeA1;
    cv::Mat planeB1;

    void dstTransform(const cv::Mat& src, cv::Mat& dest, bool invert) {
        const int flag = invert ? cv::DFT_ROWS + cv::DFT_SCALE + cv::DFT_INVERSE : cv::DFT_ROWS;
        const int rows = src.rows;
        const int cols = src.cols;

        const cv::Size sizeA(2 * cols + 2, rows);
        if (tempA.size() != sizeA) {
            tempA.create(rows, 2 * cols + 2, CV_32F);
            planeA1.create(rows, 2 * cols + 2, CV_32F);
            complexA.create(sizeA, CV_32FC2);
        }
        const cv::Size sizeB(2 * rows + 2, cols);
        if (tempB.size() != sizeB) {
            tempB.create(cols, 2 * rows + 2, CV_32F);
            planeB1.create(sizeB, CV_32F);
            complexB.create(sizeB, CV_32FC2);
        }

        tempA.setTo(0);
        src.copyTo(tempA(cv::Rect(1, 0, cols, rows)));

        for (int j = 0; j < rows; ++j) {
            neonFlipNegateRow(src.ptr<float>(j), tempA.ptr<float>(j), cols);
        }

        planeA1.setTo(0);
        cv::Mat planesA[] = {tempA, planeA1};
        cv::merge(planesA, 2, complexA);
        cv::dft(complexA, complexA, flag);
        cv::split(complexA, planesA);

        tempB.setTo(0);
        for (int j = 0; j < cols; ++j) {
            float* tempLinePtr = tempB.ptr<float>(j);
            for (int i = 0; i < rows; ++i) {
                const float val = planesA[1].ptr<float>(i)[j + 1];
                tempLinePtr[i + 1] = val;
                tempLinePtr[tempB.cols - 1 - i] = -val;
            }
        }

        planeB1.setTo(0);
        cv::Mat planesB[] = {tempB, planeB1};
        cv::merge(planesB, 2, complexB);
        cv::dft(complexB, complexB, flag);
        cv::split(complexB, planesB);

        cv::Mat transposed = planesB[1].t();
        transposed(cv::Rect(0, 1, cols, rows)).copyTo(dest);
    }
};

struct ChannelScratch {
    cv::Mat bound;
    cv::Mat boundaryPoints;
    cv::Mat lap;
    cv::Mat res;
    DstScratch dst;

    void ensure(int w, int h) {
        bound.create(h, w, CV_8UC1);
        boundaryPoints.create(h, w, CV_32F);
        lap.create(h, w, CV_32F);
        res.create(h - 2, w - 2, CV_32F);
    }
};

struct CloningWorkspace {
    cv::Size size;
    EigenFilters filters;
    cv::Mat destinationGradientX;
    cv::Mat destinationGradientY;
    cv::Mat patchGradientX;
    cv::Mat patchGradientY;
    cv::Mat binaryMaskFloat;
    cv::Mat binaryMaskFloatInverted;
    cv::Mat laplacianX;
    cv::Mat laplacianY;
    std::vector<cv::Mat> rgbxChannel;
    std::vector<cv::Mat> rgbyChannel;
    std::vector<cv::Mat> output;
    ChannelScratch channel[3];

    void ensure(const cv::Size& s) {
        if (s == size && !destinationGradientX.empty()) {
            return;
        }
        size = s;
        filters = makeFilters(s.width, s.height);
        destinationGradientX.create(s, CV_32FC3);
        destinationGradientY.create(s, CV_32FC3);
        patchGradientX.create(s, CV_32FC3);
        patchGradientY.create(s, CV_32FC3);
        binaryMaskFloat.create(s, CV_32FC1);
        binaryMaskFloatInverted.create(s, CV_32FC1);
        laplacianX.create(s, CV_32FC3);
        laplacianY.create(s, CV_32FC3);
        rgbxChannel.resize(3);
        rgbyChannel.resize(3);
        output.resize(3);
        for (int c = 0; c < 3; ++c) {
            rgbxChannel[c].create(s, CV_32F);
            rgbyChannel[c].create(s, CV_32F);
            output[c].create(s, CV_8UC1);
            channel[c].ensure(s.width, s.height);
        }
    }
};

class Cloning {
public:
    explicit Cloning(CloningWorkspace& ws) : ws_(ws) {}

    void normalClone(cv::Mat& destination, const cv::Mat& patch, cv::Mat& binaryMask, cv::Mat& cloned,
                     int flag) {
        ws_.ensure(destination.size());
        computeDerivatives(destination, patch, binaryMask);

        if (flag == kNormalClone) {
            arrayProduct(ws_.patchGradientX, ws_.binaryMaskFloat, ws_.patchGradientX);
            arrayProduct(ws_.patchGradientY, ws_.binaryMaskFloat, ws_.patchGradientY);
        } else {
            throw std::invalid_argument("seamless_clone_fft: only NORMAL_CLONE is implemented");
        }

        evaluate(destination, binaryMask, cloned);
    }

private:
    CloningWorkspace& ws_;

    static void computeGradientX(const cv::Mat& img, cv::Mat& gx) {
        const cv::Mat kernel = (cv::Mat_<char>(1, 3) << 0, -1, 1);
        cv::filter2D(img, gx, CV_32F, kernel);
        if (img.channels() == 1) {
            cv::cvtColor(gx, gx, cv::COLOR_GRAY2BGR);
        }
    }

    static void computeGradientY(const cv::Mat& img, cv::Mat& gy) {
        const cv::Mat kernel = (cv::Mat_<char>(3, 1) << 0, -1, 1);
        cv::filter2D(img, gy, CV_32F, kernel);
        if (img.channels() == 1) {
            cv::cvtColor(gy, gy, cv::COLOR_GRAY2BGR);
        }
    }

    static void computeLaplacianX(const cv::Mat& img, cv::Mat& laplacianX) {
        const cv::Mat kernel = (cv::Mat_<char>(1, 3) << -1, 1, 0);
        cv::filter2D(img, laplacianX, CV_32F, kernel);
    }

    static void computeLaplacianY(const cv::Mat& img, cv::Mat& laplacianY) {
        const cv::Mat kernel = (cv::Mat_<char>(3, 1) << -1, 1, 0);
        cv::filter2D(img, laplacianY, CV_32F, kernel);
    }

    void solvePoisson(const cv::Mat& img, cv::Mat& modDiff, cv::Mat& result, ChannelScratch& scratch) {
        const int w = img.cols;
        const int h = img.rows;

        scratch.dst.dstTransform(modDiff, scratch.res, false);

        const float* fx = ws_.filters.x.data();
        for (int j = 0; j < h - 2; ++j) {
            neonDivideEigenRow(scratch.res.ptr<float>(j), w - 2, ws_.filters.y[static_cast<size_t>(j)], fx);
        }

        scratch.dst.dstTransform(scratch.res, modDiff, true);

        for (int i = 0; i < w; ++i) {
            result.ptr<uchar>(0)[i] = img.ptr<uchar>(0)[i];
        }

        for (int j = 1; j < h - 1; ++j) {
            uchar* resLinePtr = result.ptr<uchar>(j);
            const uchar* imgLinePtr = img.ptr<uchar>(j);
            const float* interpLinePtr = modDiff.ptr<float>(j - 1);

            resLinePtr[0] = imgLinePtr[0];
            neonClampU8Row(resLinePtr, interpLinePtr, w);
            resLinePtr[w - 1] = imgLinePtr[w - 1];
        }

        uchar* lastRow = result.ptr<uchar>(h - 1);
        const uchar* imgLast = img.ptr<uchar>(h - 1);
        for (int i = 0; i < w; ++i) {
            lastRow[i] = imgLast[i];
        }
    }

    void poissonSolver(const cv::Mat& img, const cv::Mat& laplacianX, const cv::Mat& laplacianY,
                       cv::Mat& result, ChannelScratch& scratch) {
        const int w = img.cols;
        const int h = img.rows;

        cv::add(laplacianX, laplacianY, scratch.lap);

        img.copyTo(scratch.bound);
        cv::rectangle(scratch.bound, cv::Point(1, 1), cv::Point(w - 2, h - 2), cv::Scalar::all(0), -1);
        cv::Laplacian(scratch.bound, scratch.boundaryPoints, CV_32F);

        cv::subtract(scratch.lap, scratch.boundaryPoints, scratch.boundaryPoints);
        cv::Mat modDiff = scratch.boundaryPoints(cv::Rect(1, 1, w - 2, h - 2));

        solvePoisson(img, modDiff, result, scratch);
    }

    void computeDerivatives(const cv::Mat& destination, const cv::Mat& patch, cv::Mat& binaryMask) {
        computeGradientX(destination, ws_.destinationGradientX);
        computeGradientY(destination, ws_.destinationGradientY);
        computeGradientX(patch, ws_.patchGradientX);
        computeGradientY(patch, ws_.patchGradientY);

        const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
        cv::erode(binaryMask, binaryMask, kernel, cv::Point(-1, -1), 3);

        binaryMask.convertTo(ws_.binaryMaskFloat, CV_32FC1, 1.0 / 255.0);
    }

    static void arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) {
        std::vector<cv::Mat> lhsChannels;
        std::vector<cv::Mat> resultChannels;
        cv::split(lhs, lhsChannels);
        cv::split(result, resultChannels);

        for (int chan = 0; chan < 3; ++chan) {
            cv::multiply(lhsChannels[chan], rhs, resultChannels[chan]);
        }

        cv::merge(resultChannels, result);
    }

    void poisson(const cv::Mat& destination) {
        cv::add(ws_.destinationGradientX, ws_.patchGradientX, ws_.laplacianX);
        cv::add(ws_.destinationGradientY, ws_.patchGradientY, ws_.laplacianY);

        computeLaplacianX(ws_.laplacianX, ws_.laplacianX);
        computeLaplacianY(ws_.laplacianY, ws_.laplacianY);

        cv::split(ws_.laplacianX, ws_.rgbxChannel);
        cv::split(ws_.laplacianY, ws_.rgbyChannel);
        cv::split(destination, ws_.output);

        // Poisson solve is per-channel; each channel has private DST buffers. Run 3 channels in
        // parallel — this is where ~90% of time is spent (cv::dft inside DST).
        const int savedThreads = cv::getNumThreads();
        const int perChannelThreads = std::max(1, savedThreads / 3);
        cv::setNumThreads(perChannelThreads);

        cv::parallel_for_(cv::Range(0, 3), [&](const cv::Range& range) {
            for (int chan = range.start; chan < range.end; ++chan) {
                poissonSolver(ws_.output[chan], ws_.rgbxChannel[chan], ws_.rgbyChannel[chan],
                              ws_.output[chan], ws_.channel[chan]);
            }
        });

        cv::setNumThreads(savedThreads);
    }

    void evaluate(const cv::Mat& destination, cv::Mat& wmask, cv::Mat& cloned) {
        cv::bitwise_not(wmask, wmask);
        wmask.convertTo(ws_.binaryMaskFloatInverted, CV_32FC1, 1.0 / 255.0);

        arrayProduct(ws_.destinationGradientX, ws_.binaryMaskFloatInverted, ws_.destinationGradientX);
        arrayProduct(ws_.destinationGradientY, ws_.binaryMaskFloatInverted, ws_.destinationGradientY);

        poisson(destination);
        cv::merge(ws_.output, cloned);
    }
};

cv::Mat checkMaskLikeOpenCV(const cv::Mat& mask, const cv::Size& size) {
    cv::Mat gray;
    if (mask.channels() > 1) {
        cv::cvtColor(mask, gray, cv::COLOR_BGRA2GRAY);
    } else if (mask.empty()) {
        gray = cv::Mat(size.height, size.width, CV_8UC1, cv::Scalar(255));
    } else {
        mask.copyTo(gray);
    }
    return gray;
}

struct TopLevelBuffers {
    cv::Mat grayMask;
    cv::Mat processedMask;
    cv::Mat destinationROI;
    cv::Mat sourceROI;
    CloningWorkspace cloning;
};

void runClone(TopLevelBuffers& buffers, const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask,
              cv::Point center, cv::Mat& output, int flags) {
    if (src.empty() || dst.empty()) {
        throw std::invalid_argument("seamless_clone_fft: empty src/dst");
    }

    cv::Mat grayMask = checkMaskLikeOpenCV(mask, src.size());
    if (buffers.grayMask.size() != grayMask.size() || buffers.grayMask.type() != grayMask.type()) {
        buffers.grayMask.create(grayMask.size(), grayMask.type());
    }
    grayMask.copyTo(buffers.grayMask);

    dst.copyTo(output);

    const cv::Mat maskInner =
        buffers.grayMask(cv::Rect(1, 1, buffers.grayMask.cols - 2, buffers.grayMask.rows - 2));
    if (buffers.processedMask.size() != buffers.grayMask.size()) {
        buffers.processedMask.create(buffers.grayMask.size(), CV_8UC1);
    }
    cv::copyMakeBorder(maskInner, buffers.processedMask, 1, 1, 1, 1, cv::BORDER_ISOLATED | cv::BORDER_CONSTANT,
                       cv::Scalar(0));

    const cv::Rect roiS = cv::boundingRect(buffers.processedMask);
    if (roiS.empty()) {
        return;
    }

    const cv::Rect roiDest(center.x - roiS.width / 2, center.y - roiS.height / 2, roiS.width, roiS.height);

    if (buffers.destinationROI.size() != roiS.size() || buffers.destinationROI.type() != dst.type()) {
        buffers.destinationROI.create(roiS.size(), dst.type());
    }
    dst(roiDest).copyTo(buffers.destinationROI);

    if (buffers.sourceROI.size() != roiS.size() || buffers.sourceROI.type() != src.type()) {
        buffers.sourceROI.create(roiS.size(), src.type());
    }
    buffers.sourceROI.setTo(0);
    src(roiS).copyTo(buffers.sourceROI, buffers.processedMask(roiS));

    cv::Mat maskROI = buffers.processedMask(roiS);
    cv::Mat recoveredROI = output(roiDest);

    Cloning cloner(buffers.cloning);
    cloner.normalClone(buffers.destinationROI, buffers.sourceROI, maskROI, recoveredROI, flags);
}

}  // namespace

namespace seamless_clone_fft {

struct Context::Impl {
    TopLevelBuffers buffers;
};

Context::Context() : impl_(new Impl()) {}
Context::~Context() {
    delete impl_;
    impl_ = nullptr;
}

void Context::seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                            cv::Mat& output, int flags) {
    runClone(impl_->buffers, src, dst, mask, center, output, flags);
}

void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags) {
    Context ctx;
    ctx.seamlessClone(src, dst, mask, center, output, flags);
}

}  // namespace seamless_clone_fft
