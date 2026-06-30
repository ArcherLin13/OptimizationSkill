#pragma once

#include <opencv2/core.hpp>

namespace sc_fft {

// In-place 1D complex FFT along each matrix row (OpenCV DFT_ROWS semantics).
// inverse=true applies DFT_INVERSE + DFT_SCALE (multiply by 1/width).
void dftRows32fc2(cv::Mat& inout, bool inverse);

}  // namespace sc_fft
