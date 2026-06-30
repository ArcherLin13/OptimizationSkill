#pragma once

#include "metrics.h"
#include "timing.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

inline void printBanner(const char* title) {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "================================================================\n";
}

inline void printSubBanner(const char* title) {
    std::cout << "\n-- " << title << " " << std::string(58 - std::strlen(title), '-') << "\n";
}

inline void printKv(const char* key, const std::string& value) {
    std::cout << "  " << std::left << std::setw(14) << key << value << "\n";
}

inline void printResultTableHeader() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n";
    std::cout << "  " << std::left << std::setw(26) << "method" << std::setw(8) << "kind"
              << std::right << std::setw(9) << "avg_ms" << std::setw(9) << "min_ms"
              << std::setw(9) << "max_ms" << std::setw(8) << "speedup" << std::setw(9) << "psnr"
              << std::setw(9) << "maxdiff" << "  " << std::left << std::setw(6) << "ok"
              << "\n";
    std::cout << "  " << std::string(26, '-') << std::string(8, '-') << std::string(9, '-')
              << std::string(9, '-') << std::string(9, '-') << std::string(8, '-')
              << std::string(9, '-') << std::string(9, '-') << "  " << std::string(6, '-')
              << "\n";
}

inline void printResultRow(const char* name, const char* kind, const TimingStats& stats,
                           double speedup, const CompareResult* cmp, bool showCompare) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::left << std::setw(26) << name << std::setw(8) << kind << std::right
              << std::setw(9) << stats.avgMs << std::setw(9) << stats.minMs << std::setw(9)
              << stats.maxMs << std::setw(7) << speedup << "x";

    if (showCompare && cmp != nullptr) {
        std::cout << std::setw(9) << cmp->psnr << std::setw(9) << cmp->maxAbsDiff << "  "
                  << std::left << std::setw(6) << (cmp->pass ? "PASS" : "FAIL");
    } else {
        std::cout << std::setw(9) << "-" << std::setw(9) << "-" << "  " << std::left
                  << std::setw(6) << "-";
    }
    std::cout << "\n";
}

inline void printNote(const char* name, const char* note) {
    std::cout << "      -> " << name << ": " << note << "\n";
}
