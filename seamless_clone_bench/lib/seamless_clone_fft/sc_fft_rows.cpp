#include "sc_fft_rows.h"

#include <algorithm>
#include <complex>

#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"

namespace sc_fft {

void dftRows32fc2(cv::Mat& inout, bool inverse) {
    CV_Assert(inout.type() == CV_32FC2);
    const int rows = inout.rows;
    const int cols = inout.cols;
    if (rows <= 0 || cols <= 0) {
        return;
    }

    auto* data = reinterpret_cast<std::complex<float>*>(inout.ptr<float>(0));
    const pocketfft::shape_t shape{static_cast<size_t>(rows), static_cast<size_t>(cols)};
    const ptrdiff_t rowStride = static_cast<ptrdiff_t>(inout.step1());
    const pocketfft::stride_t stride{rowStride, 1};
    const pocketfft::shape_t axes{1};
    const bool forward = !inverse;
    const float fct = inverse ? (1.0f / static_cast<float>(cols)) : 1.0f;
    const size_t nthreads = static_cast<size_t>(std::max(1, cv::getNumThreads()));

    pocketfft::c2c(shape, stride, stride, axes, forward, data, data, fct, nthreads);
}

}  // namespace sc_fft
