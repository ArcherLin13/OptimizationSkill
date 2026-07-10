// HarmonyOS device test: production path (opt_2d -> probs -> CPU decode) vs
// fused (softmax+argmax -> light CPU decode).
//
// Usage: ./ocl_test_ctc [--data testdata] [--local-char 512] [--runs N] [--warmup N]

#include <CL/cl.h>
#ifdef OCR_OPENCL_DLOPEN
#include "opencl_dynload.h"
#endif

#include "ctc_decode.h"

#include <chrono>
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

#define OCL_CHECK(call, msg)                                                                  \
    do {                                                                                      \
        cl_int _err = (call);                                                                 \
        if (_err != CL_SUCCESS) {                                                             \
            std::fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", _err, __FILE__, __LINE__, \
                         msg);                                                                \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Args {
    std::string data_dir = "testdata";
    int local_char = 512;
    int runs = 20;
    int warmup = 3;
};

struct Launch2D {
    size_t global[2];
    size_t local[2];
    size_t local_mem_bytes;
};

struct PipelineTiming {
    double kernel_ms = 0.0;
    double read_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms() const { return kernel_ms + read_ms + decode_ms; }
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
                "  original = softmax_ocr_opt_2d + read probs + CPU decodeText\n"
                "  fused      = softmax_ocr_fused_ctc + read argmax + CPU decodeText\n",
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

double benchKernel2D(cl_command_queue queue, cl_kernel kernel, const Launch2D& launch, int warmup,
                     int runs) {
    for (int i = 0; i < warmup; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, launch.global, launch.local, 0,
                                         nullptr, &ev),
                  "warmup");
        OCL_CHECK(clWaitForEvents(1, &ev), "warmup wait");
        clReleaseEvent(ev);
    }
    double sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, launch.global, launch.local, 0,
                                         nullptr, &ev),
                  "kernel");
        sum += profileMs(ev);
        clReleaseEvent(ev);
    }
    return sum / runs;
}

double benchKernel1D(cl_command_queue queue, cl_kernel kernel, int warmup, int runs) {
    const size_t global = static_cast<size_t>(kSeqLen);
    for (int i = 0; i < warmup; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &ev),
                  "warmup");
        OCL_CHECK(clWaitForEvents(1, &ev), "warmup wait");
        clReleaseEvent(ev);
    }
    double sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &ev),
                  "kernel");
        sum += profileMs(ev);
        clReleaseEvent(ev);
    }
    return sum / runs;
}

PipelineTiming runOriginalPath2D(cl_command_queue queue, cl_kernel softmax_2d_k, const Launch2D& launch,
                                 cl_mem probs_buf, int warmup, int runs) {
    PipelineTiming t;
    t.kernel_ms = benchKernel2D(queue, softmax_2d_k, launch, warmup, runs);

    std::vector<float> probs(kNumElem);
    const auto t_read0 = Clock::now();
    for (int i = 0; i < runs; ++i) {
        OCL_CHECK(clEnqueueReadBuffer(queue, probs_buf, CL_TRUE, 0, kNumElem * sizeof(float),
                                      probs.data(), 0, nullptr, nullptr),
                  "read probs");
    }
    t.read_ms = msSince(t_read0) / runs;

    const auto t_dec0 = Clock::now();
    ctc::DecodeResult result;
    for (int i = 0; i < runs; ++i) {
        result = ctc::decodeTextFromProbs(probs.data(), kSeqLen, kCharSize);
    }
    t.decode_ms = msSince(t_dec0) / runs;
    (void)result;
    return t;
}

PipelineTiming runFusedPath(cl_command_queue queue, cl_kernel fused_k, cl_mem token_buf,
                            cl_mem maxprob_buf, int warmup, int runs) {
    PipelineTiming t;
    t.kernel_ms = benchKernel1D(queue, fused_k, warmup, runs);

    std::vector<int> token_ids(kSeqLen);
    std::vector<float> max_probs(kSeqLen);
    const auto t_read0 = Clock::now();
    for (int i = 0; i < runs; ++i) {
        OCL_CHECK(clEnqueueReadBuffer(queue, token_buf, CL_TRUE, 0, kSeqLen * sizeof(int),
                                      token_ids.data(), 0, nullptr, nullptr),
                  "read token_ids");
        OCL_CHECK(clEnqueueReadBuffer(queue, maxprob_buf, CL_TRUE, 0, kSeqLen * sizeof(float),
                                      max_probs.data(), 0, nullptr, nullptr),
                  "read max_probs");
    }
    t.read_ms = msSince(t_read0) / runs;

    const auto t_dec0 = Clock::now();
    ctc::DecodeResult result;
    for (int i = 0; i < runs; ++i) {
        result = ctc::decodeTextFromArgmax(token_ids.data(), max_probs.data(), kSeqLen);
    }
    t.decode_ms = msSince(t_dec0) / runs;
    (void)result;
    return t;
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
    if (logits_bytes.size() != kNumElem * sizeof(float)) {
        std::fprintf(stderr, "Bad logits.bin size\n");
        return 1;
    }
    const float* logits_host = reinterpret_cast<const float*>(logits_bytes.data());

    const std::string src = readText("softmax_ocr_opt_2d.cl") + "\n" + readText("softmax_ocr_fused_ctc.cl");
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

    const std::string build_opts = "-DLOCAL_CHAR=" + std::to_string(args.local_char);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");
    OCL_CHECK(clBuildProgram(prog, 1, &dev, build_opts.c_str(), nullptr, nullptr), "clBuildProgram");

    cl_mem logits_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kNumElem * sizeof(float),
                       const_cast<float*>(logits_host), &err);
    OCL_CHECK(err, "logits_buf");
    cl_mem probs_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, kNumElem * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "probs_buf");
    cl_mem token_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, kSeqLen * sizeof(int), nullptr, &err);
    OCL_CHECK(err, "token_buf");
    cl_mem maxprob_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, kSeqLen * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "maxprob_buf");

    const int seqlen = kSeqLen;
    const int char_size = kCharSize;
    const size_t lc = static_cast<size_t>(args.local_char);
    const Launch2D launch2d = {{static_cast<size_t>(kSeqLen), lc}, {1, lc}, lc * sizeof(float)};

    cl_kernel softmax_2d_k = clCreateKernel(prog, "softmax_ocr_opt_2d", &err);
    OCL_CHECK(err, "softmax_ocr_opt_2d");
    OCL_CHECK(clSetKernelArg(softmax_2d_k, 0, sizeof(cl_mem), &logits_buf), "2d arg0");
    OCL_CHECK(clSetKernelArg(softmax_2d_k, 1, sizeof(cl_mem), &probs_buf), "2d arg1");
    OCL_CHECK(clSetKernelArg(softmax_2d_k, 2, sizeof(seqlen), &seqlen), "2d arg2");
    OCL_CHECK(clSetKernelArg(softmax_2d_k, 3, sizeof(char_size), &char_size), "2d arg3");
    OCL_CHECK(clSetKernelArg(softmax_2d_k, 4, launch2d.local_mem_bytes, nullptr), "2d arg4 local");

    cl_kernel fused_k = clCreateKernel(prog, "softmax_ocr_fused_ctc", &err);
    OCL_CHECK(err, "softmax_ocr_fused_ctc");
    OCL_CHECK(clSetKernelArg(fused_k, 0, sizeof(cl_mem), &logits_buf), "f arg0");
    OCL_CHECK(clSetKernelArg(fused_k, 1, sizeof(cl_mem), &token_buf), "f arg1");
    OCL_CHECK(clSetKernelArg(fused_k, 2, sizeof(cl_mem), &maxprob_buf), "f arg2");
    OCL_CHECK(clSetKernelArg(fused_k, 3, sizeof(seqlen), &seqlen), "f arg3");
    OCL_CHECK(clSetKernelArg(fused_k, 4, sizeof(char_size), &char_size), "f arg4");

    // --- correctness (single run) ---
    cl_event ev = nullptr;
    OCL_CHECK(clEnqueueNDRangeKernel(queue, softmax_2d_k, 2, nullptr, launch2d.global, launch2d.local, 0,
                                     nullptr, &ev),
              "opt_2d once");
    OCL_CHECK(clWaitForEvents(1, &ev), "wait opt_2d");
    clReleaseEvent(ev);

    std::vector<float> probs(kNumElem);
    OCL_CHECK(clEnqueueReadBuffer(queue, probs_buf, CL_TRUE, 0, kNumElem * sizeof(float), probs.data(), 0,
                                  nullptr, nullptr),
              "read probs once");

    const size_t global1d = static_cast<size_t>(kSeqLen);
    OCL_CHECK(clEnqueueNDRangeKernel(queue, fused_k, 1, nullptr, &global1d, nullptr, 0, nullptr, &ev),
              "fused once");
    OCL_CHECK(clWaitForEvents(1, &ev), "wait fused");
    clReleaseEvent(ev);

    std::vector<int> token_ids(kSeqLen);
    std::vector<float> max_probs(kSeqLen);
    OCL_CHECK(clEnqueueReadBuffer(queue, token_buf, CL_TRUE, 0, kSeqLen * sizeof(int), token_ids.data(), 0,
                                  nullptr, nullptr),
              "read token once");
    OCL_CHECK(clEnqueueReadBuffer(queue, maxprob_buf, CL_TRUE, 0, kSeqLen * sizeof(float), max_probs.data(),
                                  0, nullptr, nullptr),
              "read maxprob once");

    const ctc::DecodeResult ref_decode = ctc::decodeTextFromProbs(probs.data(), kSeqLen, kCharSize);
    const ctc::DecodeResult fused_decode =
        ctc::decodeTextFromArgmax(token_ids.data(), max_probs.data(), kSeqLen);
    const bool decode_ok = ctc::decodeResultsEqual(ref_decode, fused_decode);

    std::printf("OCR CTC pipeline device test (opt_2d+decode vs fused+decode)\n");
    std::printf("  device: %s\n", deviceName(dev).c_str());
    std::printf("  seqlen=%d char_size=%d local_char=%d\n", kSeqLen, kCharSize, args.local_char);
    std::printf("  original kernel: softmax_ocr_opt_2d  global={%d,%zu} local={1,%zu}\n", kSeqLen, lc, lc);
    std::printf("  fused kernel:    softmax_ocr_fused_ctc  global={%d}\n", kSeqLen);
    std::printf("  probs readback: %.2f MB\n", static_cast<double>(kNumElem * sizeof(float)) / (1024.0 * 1024.0));
    std::printf("  fused readback: %.2f KB\n",
                static_cast<double>(kSeqLen * (sizeof(int) + sizeof(float))) / 1024.0);
    std::printf("  decode correctness: %s (emitted=%zu tokens)\n", decode_ok ? "PASS" : "FAIL",
                ref_decode.token_preds.size());
    std::printf("  warmup=%d runs=%d\n\n", args.warmup, args.runs);

    const PipelineTiming orig =
        runOriginalPath2D(queue, softmax_2d_k, launch2d, probs_buf, args.warmup, args.runs);
    const PipelineTiming fused =
        runFusedPath(queue, fused_k, token_buf, maxprob_buf, args.warmup, args.runs);

    std::printf("%-14s %10s %10s %10s %10s\n", "path", "kernel", "read", "decode", "total");
    std::printf("%-14s %9.3f ms %9.3f ms %9.3f ms %9.3f ms\n", "opt_2d+decode", orig.kernel_ms,
                orig.read_ms, orig.decode_ms, orig.total_ms());
    std::printf("%-14s %9.3f ms %9.3f ms %9.3f ms %9.3f ms\n", "fused+decode", fused.kernel_ms,
                fused.read_ms, fused.decode_ms, fused.total_ms());
    std::printf("\n=== speedup (fused vs opt_2d production path) ===\n");
    std::printf("  e2e total:     %.2fx (%.3f -> %.3f ms)\n", orig.total_ms() / fused.total_ms(),
                orig.total_ms(), fused.total_ms());
    std::printf("  kernel only:   %.2fx (%.3f -> %.3f ms)\n", orig.kernel_ms / fused.kernel_ms,
                orig.kernel_ms, fused.kernel_ms);
    std::printf("  readback:      %.3f -> %.3f ms\n", orig.read_ms, fused.read_ms);
    std::printf("  decode (CPU):  %.3f -> %.3f ms\n", orig.decode_ms, fused.decode_ms);

    clReleaseKernel(softmax_2d_k);
    clReleaseKernel(fused_k);
    clReleaseMemObject(logits_buf);
    clReleaseMemObject(probs_buf);
    clReleaseMemObject(token_buf);
    clReleaseMemObject(maxprob_buf);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return decode_ok ? 0 : 1;
}
