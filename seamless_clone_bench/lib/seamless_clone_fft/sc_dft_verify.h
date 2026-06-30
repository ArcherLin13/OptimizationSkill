#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace sc_fft {

struct DftVerifyResult {
    int length = 0;
    bool inverse = false;
    bool useSimd = false;
    double maxAbsDiff = 0.0;
    bool pass = false;
};

// Compare sc_ocv_dft row DFT vs cv::dft on random CV_32FC2 data (one row).
DftVerifyResult verifyRowDft(int len, bool inverse, bool useSimd, unsigned seed = 42);

// Log summary for DST-critical lengths (1434 fwd, 228 fwd/inv).
void logDstDftVerify();

}  // namespace sc_fft
