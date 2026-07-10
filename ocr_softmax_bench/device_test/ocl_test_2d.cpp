// Device test: baseline vs 1D opt vs 2D opt — correctness + OpenCL profiling.
// Build: OpenCL headers in third_party/; OHOS uses dlopen libOpenCL.so.

#include <CL/cl.h>
#ifdef OCR_OPENCL_DLOPEN
#include "opencl_dynload.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kSeqLen = 128;
constexpr int kCharSize = 9973;
constexpr size_t kNumElem = static_cast<size_t>(kSeqLen) * static_cast<size_t>(kCharSize);
constexpr float kTol = 1e-3f;

#define OCL_CHECK(call, msg)                                                                  \
    do {                                                                                      \
        cl_int _err = (call);                                                                 \
        if (_err != CL_SUCCESS) {                                                             \
            std::fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", _err, __FILE__, __LINE__, \
                         msg);                                                                \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

struct Args {
    std::string data_dir = "testdata";
    int local_char = 512;
    int runs = 20;
    int warmup = 3;
};

struct BenchCase {
    const char* name;
    const char* kernel;
    int ndim;
    size_t global[2];
    size_t local[2];  // local[0]=0 means nullptr
    bool use_local_arg;
    size_t local_mem_bytes;
};

struct BenchResult {
    const char* name;
    double ms;
    float max_diff;
    int rows_bad;
    bool ok;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            a.data_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--local-char") == 0 && i + 1 < argc) {
            a.local_char = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            a.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: %s [--data DIR] [--local-char N] [--runs N] [--warmup N]\n"
                "  Compares baseline / opt_1d / opt_2d on device with CL profiling.\n",
                argv[0]);
            std::exit(0);
        }
    }
    return a;
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        std::exit(1);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string readText(const std::string& path) {
    const auto bytes = readFile(path);
    return std::string(bytes.begin(), bytes.end());
}

cl_device_id pickDevice(cl_platform_id platform) {
    cl_device_id dev = nullptr;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS) {
        OCL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, nullptr), "clGetDeviceIDs");
    }
    return dev;
}

std::string deviceName(cl_device_id dev) {
    char buf[256] = {};
    OCL_CHECK(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, nullptr), "CL_DEVICE_NAME");
    return buf;
}

double profileMs(cl_event ev) {
    cl_ulong t0 = 0, t1 = 0;
    OCL_CHECK(clWaitForEvents(1, &ev), "clWaitForEvents");
    OCL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr), "START");
    OCL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr), "END");
    return static_cast<double>(t1 - t0) / 1e6;
}

float maxAbsDiff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

int badRowSums(const float* probs, float tol) {
    int bad = 0;
    for (int j = 0; j < kSeqLen; ++j) {
        double s = 0.0;
        const size_t off = static_cast<size_t>(j) * kCharSize;
        for (int k = 0; k < kCharSize; ++k) {
            s += probs[off + k];
        }
        if (std::fabs(s - 1.0) > tol) {
            ++bad;
        }
    }
    return bad;
}

std::string loadAllKernels() {
    return readText("softmax_ocr_baseline.cl") + "\n" + readText("softmax_ocr_opt.cl") + "\n" +
           readText("softmax_ocr_opt_2d.cl");
}

BenchResult runCase(cl_context ctx, cl_command_queue queue, cl_device_id dev, cl_program prog,
                    cl_mem logits_buf, cl_mem probs_buf, const BenchCase& bc, const float* ref,
                    int warmup, int runs) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(prog, bc.kernel, &err);
    OCL_CHECK(err, bc.kernel);

    const int seqlen = kSeqLen;
    const int char_size = kCharSize;
    OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &logits_buf), "arg0");
    OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &probs_buf), "arg1");
    OCL_CHECK(clSetKernelArg(kernel, 2, sizeof(seqlen), &seqlen), "arg2");
    OCL_CHECK(clSetKernelArg(kernel, 3, sizeof(char_size), &char_size), "arg3");
    if (bc.use_local_arg) {
        OCL_CHECK(clSetKernelArg(kernel, 4, bc.local_mem_bytes, nullptr), "arg4 local");
    }

    const size_t* local_ptr = bc.local[0] ? bc.local : nullptr;

    for (int i = 0; i < warmup; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, bc.ndim, nullptr, bc.global, local_ptr, 0,
                                         nullptr, &ev),
                  "warmup");
        OCL_CHECK(clWaitForEvents(1, &ev), "warmup wait");
        clReleaseEvent(ev);
    }

    double prof_sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, bc.ndim, nullptr, bc.global, local_ptr, 0,
                                         nullptr, &ev),
                  "kernel");
        prof_sum += profileMs(ev);
        clReleaseEvent(ev);
    }

    std::vector<float> out(kNumElem);
    OCL_CHECK(clEnqueueReadBuffer(queue, probs_buf, CL_TRUE, 0, kNumElem * sizeof(float),
                                  out.data(), 0, nullptr, nullptr),
              "read");

    const float diff = maxAbsDiff(out.data(), ref, kNumElem);
    const int row_bad = badRowSums(out.data(), kTol);
    clReleaseKernel(kernel);

    return BenchResult{bc.name, prof_sum / runs, diff, row_bad, diff < kTol && row_bad == 0};
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

#ifdef OCR_OPENCL_DLOPEN
    if (!opencl_load()) {
        return 1;
    }
#endif

    if ((args.local_char & (args.local_char - 1)) != 0 || args.local_char <= 0) {
        std::fprintf(stderr, "--local-char must be power of two (128/256/512)\n");
        return 1;
    }

    const auto logits_bytes = readFile(args.data_dir + "/logits.bin");
    const auto ref_bytes = readFile(args.data_dir + "/probs_ref.bin");
    if (logits_bytes.size() != kNumElem * sizeof(float) ||
        ref_bytes.size() != kNumElem * sizeof(float)) {
        std::fprintf(stderr, "Bad testdata size\n");
        return 1;
    }

    const float* logits_host = reinterpret_cast<const float*>(logits_bytes.data());
    const float* ref_host = reinterpret_cast<const float*>(ref_bytes.data());

    const std::string src = loadAllKernels();
    const char* src_ptr = src.c_str();
    const size_t src_len = src.size();

    cl_platform_id platform = nullptr;
    OCL_CHECK(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    cl_device_id dev = pickDevice(platform);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    OCL_CHECK(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    OCL_CHECK(err, "clCreateCommandQueue");

    std::string build_opts = "-DLOCAL_CHAR=" + std::to_string(args.local_char);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");
    OCL_CHECK(clBuildProgram(prog, 1, &dev, build_opts.c_str(), nullptr, nullptr), "clBuildProgram");

    cl_mem logits_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kNumElem * sizeof(float),
                       const_cast<float*>(logits_host), &err);
    OCL_CHECK(err, "logits_buf");
    cl_mem probs_buf =
        clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, kNumElem * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "probs_buf");

    const size_t lc = static_cast<size_t>(args.local_char);
    const BenchCase cases[] = {
        {"baseline (2x exp, 1D)", "softmax_ocr_baseline", 1, {static_cast<size_t>(kSeqLen), 0},
         {0, 0}, false, 0},
        {"opt_1d (1x exp, 1D)", "softmax_ocr_opt", 1, {static_cast<size_t>(kSeqLen), 0}, {0, 0},
         false, 0},
        {"opt_2d (1x exp, gy parallel)", "softmax_ocr_opt_2d", 2,
         {static_cast<size_t>(kSeqLen), lc}, {1, lc}, true, lc * sizeof(float)},
    };

    std::printf("OCR softmax device compare (baseline vs 1D vs 2D)\n");
    std::printf("  device: %s\n", deviceName(dev).c_str());
    std::printf("  seqlen=%d char_size=%d local_char=%zu\n", kSeqLen, kCharSize, lc);
    std::printf("  warmup=%d runs=%d\n\n", args.warmup, args.runs);

    std::vector<BenchResult> results;
    for (const auto& c : cases) {
        results.push_back(runCase(ctx, queue, dev, prog, logits_buf, probs_buf, c, ref_host,
                                  args.warmup, args.runs));
    }

    std::printf("%-32s %10s  %12s  %s\n", "variant", "avg_ms", "max|diff|", "check");
    std::printf("%-32s %10s  %12s  %s\n", "-------", "------", "---------", "-----");
    for (const auto& r : results) {
        std::printf("%-32s %9.3f ms  %12.3e  %s\n", r.name, r.ms, r.max_diff,
                    r.ok ? "PASS" : "FAIL");
    }

    const double base_ms = results[0].ms;
    const double opt1_ms = results[1].ms;
    const double opt2_ms = results[2].ms;
    std::printf("\n=== speedup ===\n");
    std::printf("  opt_1d vs baseline:  %.2fx  (%.3f -> %.3f ms)\n", base_ms / opt1_ms, base_ms,
                opt1_ms);
    std::printf("  opt_2d vs baseline:  %.2fx  (%.3f -> %.3f ms)\n", base_ms / opt2_ms, base_ms,
                opt2_ms);
    std::printf("  opt_2d vs opt_1d:    %.2fx  (%.3f -> %.3f ms)\n", opt1_ms / opt2_ms, opt1_ms,
                opt2_ms);

    clReleaseMemObject(logits_buf);
    clReleaseMemObject(probs_buf);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    bool all_ok = true;
    for (const auto& r : results) {
        all_ok = all_ok && r.ok;
    }
    return all_ok ? 0 : 1;
}
