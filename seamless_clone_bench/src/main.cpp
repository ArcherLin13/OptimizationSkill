#include "bench_case.h"
#include "visual_export.h"

#include <iostream>
#include <string>

int runBenchmark(const BenchCase& bench);

namespace {

struct Options {
    bool runBench = true;
    bool runExport = true;
    std::string exportDir = "./out";
    std::string imageDir;
    std::string dumpDir;
    std::string caseName = "visual";
};

void printUsage() {
    std::cout << "Usage:\n"
              << "  seamless_clone_bench                     # benchmark + save images (default)\n"
              << "  seamless_clone_bench --case text         # text-heavy synthetic inputs\n"
              << "  seamless_clone_bench --bench-only        # timing only, no images\n"
              << "  seamless_clone_bench --export [DIR]      # benchmark + save images (default: ./out)\n"
              << "  seamless_clone_bench --visual [DIR]      # save images only (default: ./out)\n"
              << "  seamless_clone_bench --images DIR ...    # use DIR/src.bmp dst.bmp mask.bmp\n"
              << "  seamless_clone_bench --dump-case DIR   # write src/dst/mask BMPs only\n"
              << "\n"
              << "Examples (run from /data/vendor/camera, images -> ./out/):\n"
              << "  cd /data/vendor/camera && ./seamless_clone_bench --case text\n"
              << "  cd /data/vendor/camera && ./seamless_clone_bench --case text --visual\n";
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
        if (arg == "--case" && i + 1 < argc) {
            opt.caseName = argv[++i];
            continue;
        }
        if (arg == "--images" && i + 1 < argc) {
            opt.imageDir = argv[++i];
            continue;
        }
        if (arg == "--dump-case" && i + 1 < argc) {
            opt.dumpDir = argv[++i];
            opt.runBench = false;
            opt.runExport = false;
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

    try {
        if (!opt.dumpDir.empty()) {
            const BenchCase bench = makeBenchCaseByName(opt.caseName);
            if (!saveBenchCaseToDir(bench, opt.dumpDir)) {
                std::cerr << "Failed to write BMPs to: " << opt.dumpDir << "\n";
                return 2;
            }
            std::cout << "Wrote src/dst/mask BMPs (" << opt.caseName << ") to: " << opt.dumpDir << "\n";
            return 0;
        }

        BenchCase bench;
        if (!opt.imageDir.empty()) {
            std::string error;
            if (!loadBenchCaseFromDir(opt.imageDir, bench, error)) {
                std::cerr << "Failed to load --images: " << error << "\n";
                return 2;
            }
            std::cout << "Loaded custom images from: " << opt.imageDir << "\n";
        } else {
            bench = makeBenchCaseByName(opt.caseName);
            std::cout << "case: " << opt.caseName << "\n";
        }

        int code = 0;
        if (opt.runBench) {
            code = runBenchmark(bench);
        }

        if (opt.runExport) {
            const int files = runVisualExport(bench, opt.exportDir);
            if (files <= 0) {
                return 3;
            }
        }

        return code;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
