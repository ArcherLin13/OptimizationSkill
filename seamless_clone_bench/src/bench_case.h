#pragma once

#include "optimized_clone.h"

#include <string>

// App-like rect mask on 729x126 with high-contrast synthetic imagery for visual review.
BenchCase makeVisualBenchCase();

// Load src.png / dst.png / mask.png from a folder (optional center.txt: "364,62").
bool loadBenchCaseFromDir(const std::string& dir, BenchCase& bench, std::string& error);

int solverBoundingRectPixels(const BenchCase& bench);
