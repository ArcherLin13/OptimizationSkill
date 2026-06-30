#pragma once

#include "optimized_clone.h"

bool isOpenCLPoissonAvailable();

bool runJacobiPoissonClone(const BenchCase& bench, cv::Mat& output, int iterations = 400,
                           bool useOpenCL = false);