#pragma once

#include "bench_case.h"
#include "seamless_clone_fft.h"

seamless_clone_fft::Context& fftPoissonContext();
bool runFftPoissonClone(const BenchCase& bench, cv::Mat& output);
