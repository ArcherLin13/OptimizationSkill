#include "sc_dft_verify.h"

#include "sc_fft_rows.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace sc_fft {

static cv::Mat makeRandomRow(int len, unsigned seed) {
    cv::Mat row(1, len, CV_32FC2);
    cv::RNG rng(static_cast<uint64>(seed));
    rng.fill(row, cv::RNG::UNIFORM, -1.0f, 1.0f);
    return row;
}

static double maxAbsDiffMat(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.size() == b.size() && a.type() == b.type());
    double maxDiff = 0.0;
    for (int y = 0; y < a.rows; ++y) {
        const auto* pa = a.ptr<cv::Vec2f>(y);
        const auto* pb = b.ptr<cv::Vec2f>(y);
        for (int x = 0; x < a.cols; ++x) {
            const double dr = std::abs(static_cast<double>(pa[x][0]) - static_cast<double>(pb[x][0]));
            const double di = std::abs(static_cast<double>(pa[x][1]) - static_cast<double>(pb[x][1]));
            maxDiff = std::max(maxDiff, std::max(dr, di));
        }
    }
    return maxDiff;
}

DftVerifyResult verifyRowDft(int len, bool inverse, bool useSimd, unsigned seed) {
    DftVerifyResult out;
    out.length = len;
    out.inverse = inverse;
    out.useSimd = useSimd;

    cv::Mat ref = makeRandomRow(len, seed);
    cv::Mat test = ref.clone();

    const int flags = inverse ? cv::DFT_ROWS + cv::DFT_SCALE + cv::DFT_INVERSE : cv::DFT_ROWS;
    cv::dft(test, test, flags);

    cv::Mat native = ref.clone();
    dftRows32fc2(native, inverse, useSimd);

    out.maxAbsDiff = maxAbsDiffMat(test, native);
    out.pass = out.maxAbsDiff <= 1e-5;
    return out;
}

void logDstDftVerify() {
    const struct Case {
        int len;
        bool inv;
    } cases[] = {{1434, false}, {228, false}, {228, true}};
    std::cout << "  [dft-verify] OpenCV cv::dft vs sc_ocv_dft row DFT (tol=1e-5)\n";
    for (const Case& c : cases) {
        for (const bool simd : {false, true}) {
            const DftVerifyResult r = verifyRowDft(c.len, c.inv, simd, 42u + static_cast<unsigned>(c.len));
            std::cout << "    len=" << r.length << " inv=" << (r.inverse ? 1 : 0)
                      << " simd=" << (r.useSimd ? 1 : 0) << " maxDiff=" << r.maxAbsDiff
                      << (r.pass ? " PASS" : " FAIL") << '\n';
        }
    }
}

}  // namespace sc_fft
