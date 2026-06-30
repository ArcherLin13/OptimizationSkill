#include "bench_case.h"
#include "visual_export.h"

#include <iostream>
#include <string>
#include <vector>

int runBenchmark();

namespace {

struct Options {
    bool runBench = true;
    bool runExport = true;
    std::string exportDir = "/data/local/tmp/seamless_clone_bench/out";
    std::string imageDir;
};

void printUsage() {
    std::cout << "Usage:\n"
              << "  seamless_clone_bench                     # benchmark + save images (default)\n"
              << "  seamless_clone_bench --bench-only        # timing only, no images\n"
              << "  seamless_clone_bench --export [DIR]      # benchmark + save images to DIR\n"
              << "  seamless_clone_bench --visual [DIR]      # save images only (no timing)\n"
              << "  seamless_clone_bench --images DIR ...    # use DIR/src.bmp dst.bmp mask.bmp\n"
              << "\n"
              << "Examples:\n"
              << "  ./seamless_clone_bench --export ./out\n"
              << "  ./seamless_clone_bench --images ./my_case --export ./out\n";
}

bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        }
        if (arg == "--export" || arg == "--visual") {
            opt.runExport = true;
            if (arg == "--visual") {
                opt.runBench = false;
            }
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opt.exportDir = argv[++i];
            }
            continue;
        }
        if (arg == "--images" && i + 1 < argc) {
            opt.imageDir = argv[++i];
            continue;
        }
        if (arg == "--bench-only") {
            opt.runBench = true;
            opt.runExport = false;
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        printUsage();
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        return 0;
    }

    BenchCase bench = makeVisualBenchCase();
    if (!opt.imageDir.empty()) {
        std::string error;
        if (!loadBenchCaseFromDir(opt.imageDir, bench, error)) {
            std::cerr << "Failed to load --images: " << error << "\n";
            return 2;
        }
        std::cout << "Loaded custom images from: " << opt.imageDir << "\n";
    }

    int code = 0;
    if (opt.runBench) {
        code = runBenchmark();
    }

    if (opt.runExport) {
        const int files = runVisualExport(bench, opt.exportDir);
        if (files <= 0) {
            return 3;
        }
    }

    return code;
}
