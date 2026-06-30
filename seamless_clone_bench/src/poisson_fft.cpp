#include "poisson_fft.h"

seamless_clone_fft::Context& fftPoissonContext() {
    static seamless_clone_fft::Context ctx;
    return ctx;
}

bool runFftPoissonClone(const BenchCase& bench, cv::Mat& output) {
    try {
        fftPoissonContext().seamlessClone(bench.src, bench.dst, bench.mask, bench.center, output, 1);
        return true;
    } catch (...) {
        return false;
    }
}

bool runFftPoissonCloneSkipUnchanged(const BenchCase& bench, cv::Mat& output) {
    static seamless_clone_fft::SkipState skip;
    try {
        seamless_clone_fft::seamlessCloneSkipUnchanged(skip, bench.src, bench.dst, bench.mask, bench.center,
                                                       output, 1, &fftPoissonContext());
        return true;
    } catch (...) {
        return false;
    }
}
