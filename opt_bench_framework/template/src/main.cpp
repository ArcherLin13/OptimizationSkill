#include "bench_case.h"

#include <iostream>
#include <string>

int runBenchmark(const BenchCase& bench);

int main(int argc, char** argv) {
    std::string caseName = "default";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: __BENCH_NAME__ [--case default|dir PATH]\n";
            return 0;
        }
        if (arg == "--case" && i + 1 < argc) {
            caseName = argv[++i];
        }
    }

    BenchCase bench = (caseName == "default") ? makeDefaultCase() : loadCaseFromDir(caseName);
    return runBenchmark(bench);
}
