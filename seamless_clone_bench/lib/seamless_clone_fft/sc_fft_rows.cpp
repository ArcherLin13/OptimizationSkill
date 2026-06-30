#include "sc_fft_rows.h"
#include "ocv_dft_32f.h"

#include <opencv2/core.hpp>

namespace sc_fft {

void dftRows32fc2(cv::Mat& inout, bool inverse, bool useSimd) {
    CV_Assert(inout.type() == CV_32FC2);
    const int cols = inout.cols;
    const bool scaled = inverse;
    for (int y = 0; y < inout.rows; ++y) {
        float* row = inout.ptr<float>(y);
        sc_ocv_dft::dft1dComplex32fInplace(row, cols, inverse, scaled, useSimd);
    }
}

}  // namespace sc_fft
