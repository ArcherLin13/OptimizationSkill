#include "poisson_fft.h"

#include "seamless_clone_fft.h"

seamless_clone_fft::Context& fftPoissonContext() {
    static seamless_clone_fft::Context ctx;
    return ctx;
}

seamless_clone_fft::Context& fftPoissonNativeContext(bool neonRadix4) {
    static seamless_clone_fft::Context ctxScalar;
    static seamless_clone_fft::Context ctxNeon;
    seamless_clone_fft::Context& ctx = neonRadix4 ? ctxNeon : ctxScalar;
    ctx.setNativeOcvDft(true, neonRadix4);
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

bool runFftPoissonNativeClone(const BenchCase& bench, cv::Mat& output, bool neonRadix4) {
    try {
        fftPoissonNativeContext(neonRadix4).seamlessClone(bench.src, bench.dst, bench.mask, bench.center,
                                                          output, 1);
        return true;
    } catch (...) {
        return false;
    }
}
