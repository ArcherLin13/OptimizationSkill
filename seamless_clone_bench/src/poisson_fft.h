#pragma once

#include "bench_case.h"
#include "seamless_clone_fft.h"

seamless_clone_fft::Context& fftPoissonContext();
seamless_clone_fft::Context& fftPoissonNativeContext(bool neonRadix4);
bool runFftPoissonClone(const BenchCase& bench, cv::Mat& output);
bool runFftPoissonNativeClone(const BenchCase& bench, cv::Mat& output, bool neonRadix4);