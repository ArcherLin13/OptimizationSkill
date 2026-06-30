#include "poisson_fft.h"

#include "seamless_clone_fft.h"



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

