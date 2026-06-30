// Ported from OpenCV 4.9 modules/photo/src/seamless_cloning*.cpp (NORMAL_CLONE path).
#include "seamless_clone_fft.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kNormalClone = 1;

class Cloning {
public:
    void normalClone(cv::Mat& destination, const cv::Mat& patch, cv::Mat& binaryMask, cv::Mat& cloned,
                     int flag) {
        computeDerivatives(destination, patch, binaryMask);

        if (flag == kNormalClone) {
            arrayProduct(patchGradientX_, binaryMaskFloat_, patchGradientX_);
            arrayProduct(patchGradientY_, binaryMaskFloat_, patchGradientY_);
        } else {
            throw std::invalid_argument("seamless_clone_fft: only NORMAL_CLONE is implemented");
        }

        evaluate(destination, binaryMask, cloned);
    }

private:
    cv::Mat destinationGradientX_;
    cv::Mat destinationGradientY_;
    cv::Mat patchGradientX_;
    cv::Mat patchGradientY_;
    cv::Mat binaryMaskFloat_;
    cv::Mat binaryMaskFloatInverted_;
    std::vector<cv::Mat> rgbxChannel_;
    std::vector<cv::Mat> rgbyChannel_;
    std::vector<cv::Mat> output_;
    std::vector<float> filterX_;
    std::vector<float> filterY_;

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

    static void dstTransform(const cv::Mat& src, cv::Mat& dest, bool invert) {
        cv::Mat temp = cv::Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

        const int flag = invert ? cv::DFT_ROWS + cv::DFT_SCALE + cv::DFT_INVERSE : cv::DFT_ROWS;

        src.copyTo(temp(cv::Rect(1, 0, src.cols, src.rows)));

        for (int j = 0; j < src.rows; ++j) {
            float* tempLinePtr = temp.ptr<float>(j);
            const float* srcLinePtr = src.ptr<float>(j);
            for (int i = 0; i < src.cols; ++i) {
                tempLinePtr[src.cols + 2 + i] = -srcLinePtr[src.cols - 1 - i];
            }
        }

        cv::Mat planes[] = {temp, cv::Mat::zeros(temp.size(), CV_32F)};
        cv::Mat complex;
        cv::merge(planes, 2, complex);
        cv::dft(complex, complex, flag);
        cv::split(complex, planes);
        temp = cv::Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

        for (int j = 0; j < src.cols; ++j) {
            float* tempLinePtr = temp.ptr<float>(j);
            for (int i = 0; i < src.rows; ++i) {
                const float val = planes[1].ptr<float>(i)[j + 1];
                tempLinePtr[i + 1] = val;
                tempLinePtr[temp.cols - 1 - i] = -val;
            }
        }

        cv::Mat planes2[] = {temp, cv::Mat::zeros(temp.size(), CV_32F)};
        cv::merge(planes2, 2, complex);
        cv::dft(complex, complex, flag);
        cv::split(complex, planes2);

        temp = planes2[1].t();
        temp(cv::Rect(0, 1, src.cols, src.rows)).copyTo(dest);
    }

    void solvePoisson(const cv::Mat& img, cv::Mat& modDiff, cv::Mat& result) {
        const int w = img.cols;
        const int h = img.rows;

        cv::Mat res;
        dstTransform(modDiff, res, false);

        for (int j = 0; j < h - 2; ++j) {
            float* resLinePtr = res.ptr<float>(j);
            for (int i = 0; i < w - 2; ++i) {
                resLinePtr[i] /= (filterX_[static_cast<size_t>(i)] + filterY_[static_cast<size_t>(j)] - 4.0f);
            }
        }

        dstTransform(res, modDiff, true);

        for (int i = 0; i < w; ++i) {
            result.ptr<uchar>(0)[i] = img.ptr<uchar>(0)[i];
        }

        for (int j = 1; j < h - 1; ++j) {
            uchar* resLinePtr = result.ptr<uchar>(j);
            const uchar* imgLinePtr = img.ptr<uchar>(j);
            const float* interpLinePtr = modDiff.ptr<float>(j - 1);

            resLinePtr[0] = imgLinePtr[0];

            for (int i = 1; i < w - 1; ++i) {
                const float value = interpLinePtr[i - 1];
                if (value < 0.f) {
                    resLinePtr[i] = 0;
                } else if (value > 255.f) {
                    resLinePtr[i] = 255;
                } else {
                    resLinePtr[i] = static_cast<uchar>(value);
                }
            }

            resLinePtr[w - 1] = imgLinePtr[w - 1];
        }

        uchar* resLinePtr = result.ptr<uchar>(h - 1);
        const uchar* imgLinePtr = img.ptr<uchar>(h - 1);
        for (int i = 0; i < w; ++i) {
            resLinePtr[i] = imgLinePtr[i];
        }
    }

    void poissonSolver(const cv::Mat& img, cv::Mat& laplacianX, cv::Mat& laplacianY, cv::Mat& result) {
        const int w = img.cols;
        const int h = img.rows;

        const cv::Mat lap = laplacianX + laplacianY;

        cv::Mat bound = img.clone();
        cv::rectangle(bound, cv::Point(1, 1), cv::Point(img.cols - 2, img.rows - 2), cv::Scalar::all(0), -1);
        cv::Mat boundaryPoints;
        cv::Laplacian(bound, boundaryPoints, CV_32F);

        const cv::Mat modSource = lap - boundaryPoints;
        cv::Mat modDiff = modSource(cv::Rect(1, 1, w - 2, h - 2));

        solvePoisson(img, modDiff, result);
    }

    void initVariables(const cv::Mat& destination) {
        destinationGradientX_ = cv::Mat(destination.size(), CV_32FC3);
        destinationGradientY_ = cv::Mat(destination.size(), CV_32FC3);
        patchGradientX_ = cv::Mat(destination.size(), CV_32FC3);
        patchGradientY_ = cv::Mat(destination.size(), CV_32FC3);
        binaryMaskFloat_ = cv::Mat(destination.size(), CV_32FC1);
        binaryMaskFloatInverted_ = cv::Mat(destination.size(), CV_32FC1);

        const int w = destination.cols;
        filterX_.resize(static_cast<size_t>(w - 2));
        double scale = CV_PI / (w - 1);
        for (int i = 0; i < w - 2; ++i) {
            filterX_[static_cast<size_t>(i)] = static_cast<float>(2.0 * std::cos(scale * (i + 1)));
        }

        const int h = destination.rows;
        filterY_.resize(static_cast<size_t>(h - 2));
        scale = CV_PI / (h - 1);
        for (int j = 0; j < h - 2; ++j) {
            filterY_[static_cast<size_t>(j)] = static_cast<float>(2.0 * std::cos(scale * (j + 1)));
        }
    }

    void computeDerivatives(const cv::Mat& destination, const cv::Mat& patch, cv::Mat& binaryMask) {
        initVariables(destination);

        computeGradientX(destination, destinationGradientX_);
        computeGradientY(destination, destinationGradientY_);
        computeGradientX(patch, patchGradientX_);
        computeGradientY(patch, patchGradientY_);

        const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
        cv::erode(binaryMask, binaryMask, kernel, cv::Point(-1, -1), 3);

        binaryMask.convertTo(binaryMaskFloat_, CV_32FC1, 1.0 / 255.0);
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
        cv::Mat laplacianX = destinationGradientX_ + patchGradientX_;
        cv::Mat laplacianY = destinationGradientY_ + patchGradientY_;

        computeLaplacianX(laplacianX, laplacianX);
        computeLaplacianY(laplacianY, laplacianY);

        cv::split(laplacianX, rgbxChannel_);
        cv::split(laplacianY, rgbyChannel_);
        cv::split(destination, output_);

        for (int chan = 0; chan < 3; ++chan) {
            poissonSolver(output_[chan], rgbxChannel_[chan], rgbyChannel_[chan], output_[chan]);
        }
    }

    void evaluate(const cv::Mat& destination, cv::Mat& wmask, cv::Mat& cloned) {
        cv::bitwise_not(wmask, wmask);
        wmask.convertTo(binaryMaskFloatInverted_, CV_32FC1, 1.0 / 255.0);

        arrayProduct(destinationGradientX_, binaryMaskFloatInverted_, destinationGradientX_);
        arrayProduct(destinationGradientY_, binaryMaskFloatInverted_, destinationGradientY_);

        poisson(destination);
        cv::merge(output_, cloned);
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

}  // namespace

namespace seamless_clone_fft {

void seamlessClone(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& mask, cv::Point center,
                   cv::Mat& output, int flags) {
    if (src.empty() || dst.empty()) {
        throw std::invalid_argument("seamless_clone_fft::seamlessClone: empty src/dst");
    }

    cv::Mat grayMask = checkMaskLikeOpenCV(mask, src.size());
    dst.copyTo(output);

    const cv::Mat maskInner = grayMask(cv::Rect(1, 1, grayMask.cols - 2, grayMask.rows - 2));
    cv::Mat processedMask;
    cv::copyMakeBorder(maskInner, processedMask, 1, 1, 1, 1, cv::BORDER_ISOLATED | cv::BORDER_CONSTANT,
                       cv::Scalar(0));

    const cv::Rect roiS = cv::boundingRect(processedMask);
    if (roiS.empty()) {
        return;
    }

    const cv::Rect roiD(center.x - roiS.width / 2, center.y - roiS.height / 2, roiS.width, roiS.height);
    cv::Mat destinationROI = dst(roiD).clone();

    cv::Mat sourceROI = cv::Mat::zeros(roiS.height, roiS.width, src.type());
    src(roiS).copyTo(sourceROI, processedMask(roiS));

    cv::Mat maskROI = processedMask(roiS);
    cv::Mat recoveredROI = output(roiD);

    Cloning cloner;
    cloner.normalClone(destinationROI, sourceROI, maskROI, recoveredROI, flags);
}

}  // namespace seamless_clone_fft
