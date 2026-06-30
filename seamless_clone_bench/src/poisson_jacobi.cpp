#include "poisson_jacobi.h"

#include "../lib/seamless_clone_jacobi/seamless_clone_jacobi.h"

bool runJacobiPoissonClone(const BenchCase& bench, cv::Mat& output, int iterations) {
    try {
        seamless_clone_jacobi::seamlessClone(bench.src, bench.dst, bench.mask, bench.center, output, 1,
                                             iterations);
        return true;
    } catch (...) {
        return false;
    }
}
