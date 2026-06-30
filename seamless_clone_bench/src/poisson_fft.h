#pragma once

#include "optimized_clone.h"

#include "../lib/seamless_clone_fft/seamless_clone_fft.h"

bool runFftPoissonClone(const BenchCase& bench, cv::Mat& output);
bool runFftPoissonCloneSkipUnchanged(const BenchCase& bench, cv::Mat& output);

seamless_clone_fft::Context& fftPoissonContext();
