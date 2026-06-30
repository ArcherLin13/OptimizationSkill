#pragma once

#include "jacobi_config.h"

#include "optimized_clone.h"

// Iterative Poisson solver (same gradient setup as OpenCV NORMAL_CLONE).
// CPU path always available; OpenCL attempted at runtime when useOpenCL=true.
bool isOpenCLPoissonAvailable();

bool runJacobiPoissonClone(const BenchCase& bench, cv::Mat& output, int iterations = kJacobiItersDefault,
                           bool useOpenCL = false);
