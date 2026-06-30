#pragma once

#include "../lib/seamless_clone_jacobi/seamless_roi.h"

#include "optimized_clone.h"

inline bool extractSeamlessRoi(const BenchCase& bench, SeamlessRoi& out) {
    return extractSeamlessRoi(bench.src, bench.dst, bench.mask, bench.center, out);
}
