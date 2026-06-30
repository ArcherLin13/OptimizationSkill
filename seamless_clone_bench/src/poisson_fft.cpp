#include "poisson_fft.h"

#include "../lib/seamless_clone_fft/seamless_clone_fft.h"

bool runFftPoissonClone(const BenchCase& bench, cv::Mat& output) {
    try {
        seamless_clone_fft::seamlessClone(bench.src, bench.dst, bench.mask, bench.center, output, 1);
        return true;
    } catch (...) {
        return false;
    }
}
