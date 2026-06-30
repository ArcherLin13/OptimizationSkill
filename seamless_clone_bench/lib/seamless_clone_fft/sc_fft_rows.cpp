#include "sc_fft_rows.h"

#include <opencv2/core.hpp>

namespace sc_fft {

void dftRows32fc2(cv::Mat& inout, bool inverse) {
    CV_Assert(inout.type() == CV_32FC2);
    const int flag = inverse ? (cv::DFT_ROWS | cv::DFT_SCALE | cv::DFT_INVERSE) : cv::DFT_ROWS;
    cv::dft(inout, inout, flag);
}

}  // namespace sc_fft
