// Device test: softmax_ocr_opt_2d correctness + OpenCL event profiling.
// Build: needs OpenCL (CL/cl.h, -lOpenCL)
// Usage:
//   ./ocl_test_2d --data testdata [--local-char 512] [--runs 20]

#include <CL/cl.h>

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
                "  Loads testdata/logits.bin, compares output to probs_ref.bin\n"
                "  Runs softmax_ocr_opt_2d with profiling (CL_QUEUE_PROFILING_ENABLE)\n",
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
    cl_int err = CL_SUCCESS;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    if (err != CL_SUCCESS) {
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
    OCL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr),
              "START");
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

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    if ((args.local_char & (args.local_char - 1)) != 0 || args.local_char <= 0) {
        std::fprintf(stderr, "--local-char must be power of two (128/256/512)\n");
        return 1;
    }

    const std::string logits_path = args.data_dir + "/logits.bin";
    const std::string ref_path = args.data_dir + "/probs_ref.bin";
    const std::string kernel_path = "softmax_ocr_opt_2d.cl";

    const auto logits_bytes = readFile(logits_path);
    const auto ref_bytes = readFile(ref_path);
    if (logits_bytes.size() != kNumElem * sizeof(float) ||
        ref_bytes.size() != kNumElem * sizeof(float)) {
        std::fprintf(stderr, "Bad testdata size (expected %zu bytes each)\n", kNumElem * sizeof(float));
        return 1;
    }

    const float* logits_host = reinterpret_cast<const float*>(logits_bytes.data());
    const float* ref_host = reinterpret_cast<const float*>(ref_bytes.data());

    const std::string kernel_src = readText(kernel_path);
    const char* src_ptr = kernel_src.c_str();
    const size_t src_len = kernel_src.size();

    cl_platform_id platform = nullptr;
    OCL_CHECK(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    cl_device_id dev = pickDevice(platform);
    const std::string dev_name = deviceName(dev);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    OCL_CHECK(err, "clCreateContext");
    cl_command_queue queue =
        clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    OCL_CHECK(err, "clCreateCommandQueue");

    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");
    std::string build_opts = "-DLOCAL_CHAR=" + std::to_string(args.local_char);
    OCL_CHECK(clBuildProgram(prog, 1, &dev, build_opts.c_str(), nullptr, nullptr),
              "clBuildProgram");

    cl_kernel kernel = clCreateKernel(prog, "softmax_ocr_opt_2d", &err);
    OCL_CHECK(err, "clCreateKernel");

    cl_mem logits_buf =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kNumElem * sizeof(float),
                       const_cast<float*>(logits_host), &err);
    OCL_CHECK(err, "logits_buf");
    cl_mem probs_buf =
        clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, kNumElem * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "probs_buf");

    const int seqlen = kSeqLen;
    const int char_size = kCharSize;
    const size_t local_mem = static_cast<size_t>(args.local_char) * sizeof(float);
    OCL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &logits_buf), "arg0");
    OCL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &probs_buf), "arg1");
    OCL_CHECK(clSetKernelArg(kernel, 2, sizeof(seqlen), &seqlen), "arg2");
    OCL_CHECK(clSetKernelArg(kernel, 3, sizeof(char_size), &char_size), "arg3");
    OCL_CHECK(clSetKernelArg(kernel, 4, local_mem, nullptr), "arg4 local");

    const size_t global[2] = {static_cast<size_t>(kSeqLen), static_cast<size_t>(args.local_char)};
    const size_t local[2] = {1, static_cast<size_t>(args.local_char)};

    std::printf("softmax_ocr_opt_2d device test\n");
    std::printf("  device: %s\n", dev_name.c_str());
    std::printf("  seqlen=%d char_size=%d local_char=%d\n", kSeqLen, kCharSize, args.local_char);
    std::printf("  global={%zu,%zu} local={%zu,%zu} local_mem=%zu B\n", global[0], global[1],
                local[0], local[1], local_mem);
    std::printf("  build: %s\n", build_opts.c_str());
    std::printf("\n");

    for (int i = 0; i < args.warmup; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, &ev),
                  "warmup kernel");
        OCL_CHECK(clWaitForEvents(1, &ev), "warmup wait");
        clReleaseEvent(ev);
    }

    double prof_sum_ms = 0.0;
    for (int i = 0; i < args.runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, &ev),
                  "kernel");
        prof_sum_ms += profileMs(ev);
        clReleaseEvent(ev);
    }
    const double prof_avg_ms = prof_sum_ms / args.runs;

    std::vector<float> out(kNumElem);
    OCL_CHECK(clEnqueueReadBuffer(queue, probs_buf, CL_TRUE, 0, kNumElem * sizeof(float),
                                  out.data(), 0, nullptr, nullptr),
              "read probs");

    const float diff = maxAbsDiff(out.data(), ref_host, kNumElem);
    const int row_bad = badRowSums(out.data(), kTol);
    const bool pass = diff < kTol && row_bad == 0;

    std::printf("=== results ===\n");
    std::printf("  correctness max|diff|=%.6e  rows_bad=%d  %s\n", diff, row_bad,
                pass ? "PASS" : "FAIL");
    std::printf("  OpenCL profiling (kernel only): avg=%.3f ms  (runs=%d)\n", prof_avg_ms, args.runs);
    std::printf("\n");
    std::printf("Note: profiling time excludes clEnqueueReadBuffer.\n");
    std::printf("      Enable CL_QUEUE_PROFILING_ENABLE on queue (already on).\n");

    clReleaseMemObject(logits_buf);
    clReleaseMemObject(probs_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return pass ? 0 : 1;
}
