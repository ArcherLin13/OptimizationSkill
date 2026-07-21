// HarmonyOS OpenCL: findMaxValue baseline bench (kernel time only).
// Image: half buffer, default 5760x4320.
//
// Usage:
//   ./ocl_test_findmax [--kernel PATH] [--width 5760] [--height 4320] [--runs 30]
//                       [--wg 256] [--lws 256]

#include <CL/cl.h>
#ifdef OCR_OPENCL_DLOPEN
#include "opencl_dynload.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

#define OCL_CHECK(call, msg)                                                                       \
    do {                                                                                           \
        cl_int _err = (call);                                                                      \
        if (_err != CL_SUCCESS) {                                                                  \
            std::fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", _err, __FILE__, __LINE__, msg); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

struct Args {
    std::string kernel_path = "findmax_baseline.cl";
    int width = 5760;
    int height = 4320;
    int runs = 30;
    int warmup = 5;
    int wg = 256;   // compile-time WG_SIZE
    int lws = 256;  // local work size (must match WG_SIZE for this kernel)
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            a.kernel_path = argv[++i];
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            a.width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            a.height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            a.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--wg") == 0 && i + 1 < argc) {
            a.wg = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lws") == 0 && i + 1 < argc) {
            a.lws = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: %s [--kernel PATH] [--width W] [--height H] [--runs N] [--wg N] [--lws N]\n"
                "  Times findMaxValue kernel only (no H2D/D2H in measured region).\n",
                argv[0]);
            std::exit(0);
        }
    }
    return a;
}

std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        std::exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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
    OCL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr),
              "START");
    OCL_CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr),
              "END");
    return static_cast<double>(t1 - t0) / 1e6;
}

// IEEE754 binary16 <-> float (host reference).
uint16_t floatToHalf(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000u;
        const uint32_t t = mant >> (1 - exp + 13);
        return static_cast<uint16_t>(sign | t);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);  // inf
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float halfToFloat(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

void fillHalfImage(std::vector<uint16_t>& img, int w, int h, float& ref_max) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    img.resize(n);
    ref_max = -1e30f;
    // Deterministic pattern; inject a known peak away from (0,0).
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = static_cast<float>(((x * 13 + y * 7) % 1000)) / 1000.f;  // [0,1)
            if (x == w * 2 / 3 && y == h * 3 / 5) {
                v = 12.5f;  // unique max
            }
            img[static_cast<size_t>(y) * w + x] = floatToHalf(v);
            ref_max = std::max(ref_max, v);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    if (args.lws != args.wg) {
        std::fprintf(stderr, "This baseline kernel requires --lws == --wg (got lws=%d wg=%d)\n",
                     args.lws, args.wg);
        return 1;
    }
    if (args.wg <= 0 || (args.wg & (args.wg - 1)) != 0) {
        std::fprintf(stderr, "--wg must be power-of-two\n");
        return 1;
    }

#ifdef OCR_OPENCL_DLOPEN
    if (!opencl_load()) {
        return 1;
    }
#endif

    const int W = args.width;
    const int H = args.height;
    const size_t nPix = static_cast<size_t>(W) * static_cast<size_t>(H);
    const size_t bytes = nPix * sizeof(uint16_t);

    std::vector<uint16_t> host;
    float ref_max = 0.f;
    fillHalfImage(host, W, H, ref_max);

    cl_uint np = 0;
    OCL_CHECK(clGetPlatformIDs(0, nullptr, &np), "platforms count");
    if (np == 0) {
        std::fprintf(stderr, "No OpenCL platforms\n");
        return 1;
    }
    std::vector<cl_platform_id> platforms(np);
    OCL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr), "platforms");
    cl_platform_id platform = platforms[0];
    cl_device_id device = pickDevice(platform);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    OCL_CHECK(err, "clCreateContext");
    cl_command_queue queue =
        clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    OCL_CHECK(err, "clCreateCommandQueue");

    const std::string src = readText(args.kernel_path);
    const char* srcp = src.c_str();
    size_t srcl = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, &srcl, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");

    char opts[128];
    std::snprintf(opts, sizeof(opts), "-cl-std=CL1.2 -DWG_SIZE=%d", args.wg);
    err = clBuildProgram(prog, 1, &device, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::fprintf(stderr, "Build failed:\n%s\n", log.data());
        return 1;
    }

    cl_kernel kn = clCreateKernel(prog, "findMaxValue", &err);
    OCL_CHECK(err, "clCreateKernel findMaxValue");

    cl_mem buf_src = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host.data(),
                                    &err);
    OCL_CHECK(err, "buf_src");
    // 4-byte buffer for atomic_max_half (half in low 16 bits).
    cl_mem buf_max = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    OCL_CHECK(err, "buf_max");

    const unsigned int wu = static_cast<unsigned int>(W);
    const unsigned int hu = static_cast<unsigned int>(H);
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_src), "arg0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(unsigned int), &wu), "arg1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(unsigned int), &hu), "arg2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &buf_max), "arg3");

    const size_t lws = static_cast<size_t>(args.lws);
    const size_t gws = ((nPix + lws - 1) / lws) * lws;

    auto resetMax = [&] {
        uint32_t init = floatToHalf(-65504.0f);  // low 16 = -inf-ish half
        OCL_CHECK(clEnqueueWriteBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &init, 0,
                                       nullptr, nullptr),
                  "reset max");
    };

    // Correctness once
    resetMax();
    {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn, 1, nullptr, &gws, &lws, 0, nullptr, &ev),
                  "enqueue correctness");
        OCL_CHECK(clWaitForEvents(1, &ev), "wait");
        clReleaseEvent(ev);
    }
    uint32_t got_bits = 0;
    OCL_CHECK(clEnqueueReadBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &got_bits, 0, nullptr,
                                  nullptr),
              "read max");
    const float got = halfToFloat(static_cast<uint16_t>(got_bits & 0xffffu));
    const float abs_err = std::fabs(got - ref_max);

    std::printf("=== findMaxValue baseline GPU bench ===\n");
    std::printf("device: %s\n", deviceName(device).c_str());
    std::printf("size:   %d x %d = %.2f MP (%.2f MB half)\n", W, H, nPix / 1e6, bytes / 1e6);
    std::printf("kernel: %s\n", args.kernel_path.c_str());
    std::printf("gws=%zu lws=%zu wg=%d\n", gws, lws, args.wg);
    std::printf("ref_max=%.6f gpu_max=%.6f abs_err=%.3e %s\n\n", ref_max, got,
                abs_err, abs_err < 1e-2f ? "OK" : "FAIL");
    if (abs_err >= 1e-2f) {
        return 1;
    }

    // Warmup
    for (int i = 0; i < args.warmup; ++i) {
        resetMax();
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn, 1, nullptr, &gws, &lws, 0, nullptr, &ev),
                  "warmup");
        OCL_CHECK(clWaitForEvents(1, &ev), "warmup wait");
        clReleaseEvent(ev);
    }

    double sum = 0.0, best = 1e300;
    for (int i = 0; i < args.runs; ++i) {
        resetMax();
        OCL_CHECK(clFinish(queue), "finish before");
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn, 1, nullptr, &gws, &lws, 0, nullptr, &ev),
                  "bench");
        const double ms = profileMs(ev);
        clReleaseEvent(ev);
        sum += ms;
        best = std::min(best, ms);
    }

    const double avg = sum / args.runs;
    const double gb = bytes / 1e9;
    std::printf("--- kernel time only (avg %d runs, reset max not included) ---\n", args.runs);
    std::printf("  avg:  %.3f ms\n", avg);
    std::printf("  best: %.3f ms\n", best);
    std::printf("  ~bandwidth (read src once): %.2f GB/s (avg)\n", gb / (avg / 1e3));

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_max);
    clReleaseKernel(kn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
