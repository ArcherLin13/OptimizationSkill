// HarmonyOS OpenCL: planar RGB -> CV_32FC3 (interleaved BGR float).
// Compares baseline (1 px/WI) vs ppx kernels (N px/WI + float4/uchar4 loads).
//
// Usage:
//   ./ocl_test_planar [--kernel PATH] [--width W] [--height H] [--runs N]

#include <CL/cl.h>
#ifdef OCR_OPENCL_DLOPEN
#include "opencl_dynload.h"
#endif
#include "neon_planar_to_cv32fc3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

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
    std::string kernel_path = "planar_rgb_to_cv32fc3.cl";
    int width = 3840;
    int height = 2160;
    int runs = 20;
    int warmup = 3;
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
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: %s [--kernel PATH] [--width W] [--height H] [--runs N]\n"
                "  Planar R|G|B -> CV_32FC3 BGR float on OpenCL GPU.\n",
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

float maxAbsDiff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

void fillPlanes(std::vector<float>& r, std::vector<float>& g, std::vector<float>& b, int w, int h) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    r.resize(n);
    g.resize(n);
    b.resize(n);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            r[i] = static_cast<float>((x * 13 + y * 7) % 256) / 255.f;
            g[i] = static_cast<float>((x * 3 + y * 17) % 256) / 255.f;
            b[i] = static_cast<float>((x * 29 + y * 5) % 256) / 255.f;
        }
    }
}

// Pure scalar reference: disable auto-vectorization so "scalar vs NEON" is fair.
// Without this, clang -O2/-O3 often turns the loop into NEON anyway → identical times.
#if defined(__clang__)
#define NO_AUTO_VEC _Pragma("clang loop vectorize(disable) interleave(disable)")
#elif defined(__GNUC__)
#define NO_AUTO_VEC _Pragma("GCC ivdep") /* weak; prefer clang for OHOS */
#else
#define NO_AUTO_VEC
#endif

void cpuRefBgrRows(const float* r, const float* g, const float* b, float* dst, int w, int y0,
                   int y1) {
    for (int y = y0; y < y1; ++y) {
        const float* rr = r + static_cast<size_t>(y) * w;
        const float* gg = g + static_cast<size_t>(y) * w;
        const float* bb = b + static_cast<size_t>(y) * w;
        float* out = dst + static_cast<size_t>(y) * w * 3;
        NO_AUTO_VEC
        for (int x = 0; x < w; ++x) {
            out[0] = bb[x];
            out[1] = gg[x];
            out[2] = rr[x];
            out += 3;
        }
    }
}

void cpuRefBgr(const float* r, const float* g, const float* b, float* dst, int w, int h) {
    cpuRefBgrRows(r, g, b, dst, w, 0, h);
}

void cpuRefBgrMt(const float* r, const float* g, const float* b, float* dst, int w, int h,
                 int threads) {
    threads = std::max(1, std::min(threads, h));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const int y0 = h * t / threads;
        const int y1 = h * (t + 1) / threads;
        workers.emplace_back([=] { cpuRefBgrRows(r, g, b, dst, w, y0, y1); });
    }
    for (auto& th : workers) {
        th.join();
    }
}

void fillUcharPlanes(std::vector<unsigned char>& r, std::vector<unsigned char>& g,
                     std::vector<unsigned char>& b, int w, int h) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    r.resize(n);
    g.resize(n);
    b.resize(n);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            r[i] = static_cast<unsigned char>((x * 13 + y * 7) % 256);
            g[i] = static_cast<unsigned char>((x * 3 + y * 17) % 256);
            b[i] = static_cast<unsigned char>((x * 29 + y * 5) % 256);
        }
    }
}

void cpuRefBgrUchar(const unsigned char* r, const unsigned char* g, const unsigned char* b,
                    float* dst, int w, int h) {
    constexpr float k = 1.f / 255.f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t si = static_cast<size_t>(y) * w + x;
            float* out = dst + si * 3;
            out[0] = b[si] * k;
            out[1] = g[si] * k;
            out[2] = r[si] * k;
        }
    }
}

cl_program buildProgram(cl_context ctx, cl_device_id dev, const std::string& src,
                        const char* options) {
    const char* csrc = src.c_str();
    size_t len = src.size();
    cl_int err = 0;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &len, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::fprintf(stderr, "Build failed (%s):\n%s\n", options ? options : "", log.data());
        std::exit(1);
    }
    return prog;
}

using Clock = std::chrono::steady_clock;

inline double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <typename Fn>
double benchCpuMs(int warmup, int runs, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    double sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = Clock::now();
        fn();
        sum += msSince(t0);
    }
    return sum / runs;
}

struct CaseResult {
    std::string name;
    float max_diff = 0.f;
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    bool ok = false;
};

double runKernelTimed(cl_command_queue q, cl_kernel kn, size_t global[2], const size_t* local,
                      int warmup, int runs) {
    for (int i = 0; i < warmup; ++i) {
        OCL_CHECK(
            clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, local, 0, nullptr, nullptr),
            "warmup");
    }
    OCL_CHECK(clFinish(q), "finish warmup");
    double sum_ms = 0.0;
    for (int i = 0; i < runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, local, 0, nullptr, &ev),
                  "enqueue");
        sum_ms += profileMs(ev);
        clReleaseEvent(ev);
    }
    return sum_ms / runs;
}

// Pad global size up so it is divisible by local.
void padGlobal(size_t global[2], const size_t local[2]) {
    for (int i = 0; i < 2; ++i) {
        if (local[i] == 0) {
            continue;
        }
        global[i] = ((global[i] + local[i] - 1) / local[i]) * local[i];
    }
}

double gbps(double bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return (bytes / 1e9) / (ms / 1e3);
}

CaseResult runFloatSeparate(cl_context ctx, cl_command_queue q, cl_program prog, const Args& args,
                            const char* kernel_name, int pixels_per_wi, const char* label,
                            const std::vector<float>& r, const std::vector<float>& g,
                            const std::vector<float>& b, const std::vector<float>& ref) {
    const int w = args.width;
    const int h = args.height;
    const size_t n = static_cast<size_t>(w) * h;
    const size_t out_n = n * 3;
    const int src_stride = w;
    const int dst_stride = w * 3;

    cl_int err = 0;
    cl_mem buf_r = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float),
                                  const_cast<float*>(r.data()), &err);
    OCL_CHECK(err, "buf_r");
    cl_mem buf_g = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float),
                                  const_cast<float*>(g.data()), &err);
    OCL_CHECK(err, "buf_g");
    cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float),
                                  const_cast<float*>(b.data()), &err);
    OCL_CHECK(err, "buf_b");
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "buf_out");

    cl_kernel kn = clCreateKernel(prog, kernel_name, &err);
    OCL_CHECK(err, kernel_name);
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_r), "arg0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &buf_g), "arg1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(cl_mem), &buf_b), "arg2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &buf_out), "arg3");
    OCL_CHECK(clSetKernelArg(kn, 4, sizeof(int), &w), "arg4");
    OCL_CHECK(clSetKernelArg(kn, 5, sizeof(int), &h), "arg5");
    OCL_CHECK(clSetKernelArg(kn, 6, sizeof(int), &src_stride), "arg6");
    OCL_CHECK(clSetKernelArg(kn, 7, sizeof(int), &dst_stride), "arg7");

    const size_t gx = (static_cast<size_t>(w) + pixels_per_wi - 1) / pixels_per_wi;
    size_t global[2] = {gx, static_cast<size_t>(h)};
    size_t local[2] = {16, 8};
    size_t global_pad[2] = {global[0], global[1]};
    padGlobal(global_pad, local);
    const double gpu_ms =
        runKernelTimed(q, kn, global_pad, local, args.warmup, args.runs);

    std::vector<float> out(out_n);
    OCL_CHECK(clEnqueueReadBuffer(q, buf_out, CL_TRUE, 0, out_n * sizeof(float), out.data(), 0,
                                  nullptr, nullptr),
              "read");
    const float diff = maxAbsDiff(out.data(), ref.data(), out_n);

    clReleaseKernel(kn);
    clReleaseMemObject(buf_r);
    clReleaseMemObject(buf_g);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_out);

    CaseResult res;
    res.name = label;
    res.max_diff = diff;
    res.gpu_ms = gpu_ms;
    res.ok = diff <= 1e-5f;
    return res;
}

// Kernel-only vs end-to-end (H2D + kernel + D2H) for one float case.
void benchFloatE2E(cl_context ctx, cl_command_queue q, cl_program prog, const Args& args,
                   const std::vector<float>& r, const std::vector<float>& g,
                   const std::vector<float>& b, double* kernel_ms, double* e2e_ms) {
    const int w = args.width;
    const int h = args.height;
    const size_t n = static_cast<size_t>(w) * h;
    const size_t out_n = n * 3;
    const int src_stride = w;
    const int dst_stride = w * 3;
    constexpr int ppx = 8;

    cl_int err = 0;
    cl_mem buf_r = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "e2e r");
    cl_mem buf_g = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "e2e g");
    cl_mem buf_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "e2e b");
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "e2e out");

    cl_kernel kn = clCreateKernel(prog, "planar_rgb_to_cv32fc3_ppx", &err);
    OCL_CHECK(err, "e2e kn");
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_r), "a0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &buf_g), "a1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(cl_mem), &buf_b), "a2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &buf_out), "a3");
    OCL_CHECK(clSetKernelArg(kn, 4, sizeof(int), &w), "a4");
    OCL_CHECK(clSetKernelArg(kn, 5, sizeof(int), &h), "a5");
    OCL_CHECK(clSetKernelArg(kn, 6, sizeof(int), &src_stride), "a6");
    OCL_CHECK(clSetKernelArg(kn, 7, sizeof(int), &dst_stride), "a7");

    size_t global[2] = {(static_cast<size_t>(w) + ppx - 1) / ppx, static_cast<size_t>(h)};
    size_t local[2] = {16, 8};
    padGlobal(global, local);

    std::vector<float> out(out_n);
    for (int i = 0; i < args.warmup; ++i) {
        OCL_CHECK(clEnqueueWriteBuffer(q, buf_r, CL_FALSE, 0, n * sizeof(float), r.data(), 0,
                                       nullptr, nullptr),
                  "w r");
        OCL_CHECK(clEnqueueWriteBuffer(q, buf_g, CL_FALSE, 0, n * sizeof(float), g.data(), 0,
                                       nullptr, nullptr),
                  "w g");
        OCL_CHECK(clEnqueueWriteBuffer(q, buf_b, CL_FALSE, 0, n * sizeof(float), b.data(), 0,
                                       nullptr, nullptr),
                  "w b");
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, local, 0, nullptr, nullptr),
                  "kn");
        OCL_CHECK(clEnqueueReadBuffer(q, buf_out, CL_TRUE, 0, out_n * sizeof(float), out.data(), 0,
                                      nullptr, nullptr),
                  "rd");
    }

    double sum_k = 0.0;
    double sum_e2e = 0.0;
    for (int i = 0; i < args.runs; ++i) {
        const auto t0 = Clock::now();
        OCL_CHECK(clEnqueueWriteBuffer(q, buf_r, CL_FALSE, 0, n * sizeof(float), r.data(), 0,
                                       nullptr, nullptr),
                  "w r");
        OCL_CHECK(clEnqueueWriteBuffer(q, buf_g, CL_FALSE, 0, n * sizeof(float), g.data(), 0,
                                       nullptr, nullptr),
                  "w g");
        OCL_CHECK(clEnqueueWriteBuffer(q, buf_b, CL_FALSE, 0, n * sizeof(float), b.data(), 0,
                                       nullptr, nullptr),
                  "w b");
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, local, 0, nullptr, &ev), "kn");
        OCL_CHECK(clEnqueueReadBuffer(q, buf_out, CL_TRUE, 0, out_n * sizeof(float), out.data(), 0,
                                      nullptr, nullptr),
                  "rd");
        sum_e2e += msSince(t0);
        sum_k += profileMs(ev);
        clReleaseEvent(ev);
    }
    *kernel_ms = sum_k / args.runs;
    *e2e_ms = sum_e2e / args.runs;

    clReleaseKernel(kn);
    clReleaseMemObject(buf_r);
    clReleaseMemObject(buf_g);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_out);
}

CaseResult runUcharSeparate(cl_context ctx, cl_command_queue q, cl_program prog, const Args& args,
                            const char* kernel_name, int pixels_per_wi, const char* label,
                            const std::vector<unsigned char>& r,
                            const std::vector<unsigned char>& g,
                            const std::vector<unsigned char>& b, const std::vector<float>& ref) {
    const int w = args.width;
    const int h = args.height;
    const size_t n = static_cast<size_t>(w) * h;
    const size_t out_n = n * 3;
    const int src_stride = w;
    const int dst_stride = w * 3;

    cl_int err = 0;
    cl_mem buf_r =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, const_cast<unsigned char*>(r.data()),
                       &err);
    OCL_CHECK(err, "buf_r u8");
    cl_mem buf_g =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, const_cast<unsigned char*>(g.data()),
                       &err);
    OCL_CHECK(err, "buf_g u8");
    cl_mem buf_b =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, const_cast<unsigned char*>(b.data()),
                       &err);
    OCL_CHECK(err, "buf_b u8");
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "buf_out u8");

    cl_kernel kn = clCreateKernel(prog, kernel_name, &err);
    OCL_CHECK(err, kernel_name);
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_r), "arg0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &buf_g), "arg1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(cl_mem), &buf_b), "arg2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &buf_out), "arg3");
    OCL_CHECK(clSetKernelArg(kn, 4, sizeof(int), &w), "arg4");
    OCL_CHECK(clSetKernelArg(kn, 5, sizeof(int), &h), "arg5");
    OCL_CHECK(clSetKernelArg(kn, 6, sizeof(int), &src_stride), "arg6");
    OCL_CHECK(clSetKernelArg(kn, 7, sizeof(int), &dst_stride), "arg7");

    const size_t gx = (static_cast<size_t>(w) + pixels_per_wi - 1) / pixels_per_wi;
    size_t global[2] = {gx, static_cast<size_t>(h)};
    size_t local[2] = {16, 8};
    padGlobal(global, local);
    const double gpu_ms = runKernelTimed(q, kn, global, local, args.warmup, args.runs);

    std::vector<float> out(out_n);
    OCL_CHECK(clEnqueueReadBuffer(q, buf_out, CL_TRUE, 0, out_n * sizeof(float), out.data(), 0,
                                  nullptr, nullptr),
              "read");
    const float diff = maxAbsDiff(out.data(), ref.data(), out_n);

    clReleaseKernel(kn);
    clReleaseMemObject(buf_r);
    clReleaseMemObject(buf_g);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_out);

    CaseResult res;
    res.name = label;
    res.max_diff = diff;
    res.gpu_ms = gpu_ms;
    res.ok = diff <= 1e-5f;
    return res;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef OCR_OPENCL_DLOPEN
    if (!opencl_load()) {
        std::fprintf(stderr, "Failed to dlopen libOpenCL.so\n");
        return 1;
    }
#endif

    const Args args = parseArgs(argc, argv);
    if (args.width <= 0 || args.height <= 0) {
        std::fprintf(stderr, "Invalid size\n");
        return 1;
    }

    const std::string src = readText(args.kernel_path);

    cl_int err = 0;
    cl_uint nplat = 0;
    OCL_CHECK(clGetPlatformIDs(0, nullptr, &nplat), "platforms");
    if (nplat == 0) {
        std::fprintf(stderr, "No OpenCL platforms\n");
        return 1;
    }
    cl_platform_id plat = nullptr;
    OCL_CHECK(clGetPlatformIDs(1, &plat, nullptr), "get platform");
    cl_device_id dev = pickDevice(plat);
    std::printf("Device: %s\n", deviceName(dev).c_str());
    std::printf("Size:   %d x %d  runs=%d\n", args.width, args.height, args.runs);
    std::printf("Kernel: %s\n\n", args.kernel_path.c_str());

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    OCL_CHECK(err, "context");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    OCL_CHECK(err, "queue");

    cl_program prog_f =
        buildProgram(ctx, dev, src, "-cl-fast-relaxed-math -DPIXELS_PER_WI=8");
    cl_program prog_f1 = buildProgram(ctx, dev, src, "-cl-fast-relaxed-math");

    std::vector<float> r, g, b;
    fillPlanes(r, g, b, args.width, args.height);
    std::vector<float> ref(static_cast<size_t>(args.width) * args.height * 3);
    cpuRefBgr(r.data(), g.data(), b.data(), ref.data(), args.width, args.height);

    const int cpu_threads =
        std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<float> cpu_dst(ref.size());
    const double cpu_1t_ms = benchCpuMs(args.warmup, args.runs, [&] {
        cpuRefBgr(r.data(), g.data(), b.data(), cpu_dst.data(), args.width, args.height);
    });
    const double cpu_mt_ms = benchCpuMs(args.warmup, args.runs, [&] {
        cpuRefBgrMt(r.data(), g.data(), b.data(), cpu_dst.data(), args.width, args.height,
                    cpu_threads);
    });

    const int src_stride = args.width;
    const int dst_stride = args.width * 3;
    std::vector<float> neon_dst(ref.size(), 0.f);
    neon_planar_rgb_f32_to_cv32fc3(r.data(), g.data(), b.data(), neon_dst.data(), args.width,
                                   args.height, src_stride, dst_stride);
    const float neon_diff = maxAbsDiff(neon_dst.data(), ref.data(), ref.size());
    const bool neon_ok = neon_diff <= 1e-5f;

    const double neon_1t_ms = benchCpuMs(args.warmup, args.runs, [&] {
        neon_planar_rgb_f32_to_cv32fc3(r.data(), g.data(), b.data(), neon_dst.data(), args.width,
                                       args.height, src_stride, dst_stride);
    });
    const double neon_mt_ms = benchCpuMs(args.warmup, args.runs, [&] {
        neon_planar_rgb_f32_to_cv32fc3_mt(r.data(), g.data(), b.data(), neon_dst.data(),
                                          args.width, args.height, src_stride, dst_stride,
                                          cpu_threads);
    });

    // Traffic: 3 planar float reads + 3-channel float write = 24 bytes / pixel
    const double bytes =
        static_cast<double>(args.width) * args.height * 3.0 * sizeof(float) * 2.0;

    CaseResult c1 = runFloatSeparate(ctx, q, prog_f1, args, "planar_rgb_to_cv32fc3", 1,
                                     "float 1px/WI", r, g, b, ref);
    CaseResult c8 = runFloatSeparate(ctx, q, prog_f, args, "planar_rgb_to_cv32fc3_ppx", 8,
                                     "float 8px/WI", r, g, b, ref);

    double e2e_kernel = 0.0;
    double e2e_wall = 0.0;
    benchFloatE2E(ctx, q, prog_f, args, r, g, b, &e2e_kernel, &e2e_wall);

    std::printf("NEON: %s\n\n", neon_planar_available() ? "enabled (AArch64 intrinsics)"
                                                         : "fallback scalar");

    std::printf("=== CPU / NEON ===\n");
    std::printf("  Note: this convert is DRAM-bound (~24 B/pix). NEON rarely helps much.\n");
    std::printf("  If scalar≈NEON before, clang likely auto-vectorized scalar already.\n");
    std::printf("  scalar below uses #pragma clang loop vectorize(disable).\n");
    std::printf("  scalar 1t:  %.3f ms  (%.1f GB/s)\n", cpu_1t_ms, gbps(bytes, cpu_1t_ms));
    std::printf("  scalar %dt: %.3f ms  (%.1f GB/s)\n", cpu_threads, cpu_mt_ms,
                gbps(bytes, cpu_mt_ms));
    std::printf("  NEON 1t:    %.3f ms  (%.1f GB/s)  max_diff=%.3e  %s  (vs scalar1t %.2fx)\n",
                neon_1t_ms, gbps(bytes, neon_1t_ms), neon_diff, neon_ok ? "OK" : "FAIL",
                cpu_1t_ms / neon_1t_ms);
    std::printf("  NEON %dt:   %.3f ms  (%.1f GB/s)  (vs scalarMT %.2fx)\n\n", cpu_threads,
                neon_mt_ms, gbps(bytes, neon_mt_ms), cpu_mt_ms / neon_mt_ms);

    std::printf("=== GPU kernel only (data already on device) ===\n");
    bool all_ok = neon_ok;
    for (const CaseResult& c : {c1, c8}) {
        std::printf(
            "  %-16s  max_diff=%.3e  gpu=%.3f ms  (%.1f GB/s)  vsNEON1t=%.2fx  vsNEONMT=%.2fx  %s\n",
            c.name.c_str(), c.max_diff, c.gpu_ms, gbps(bytes, c.gpu_ms), neon_1t_ms / c.gpu_ms,
            neon_mt_ms / c.gpu_ms, c.ok ? "OK" : "FAIL");
        all_ok = all_ok && c.ok;
    }

    std::printf("\n=== GPU end-to-end (H2D + kernel + D2H)  float 8px/WI ===\n");
    std::printf("  kernel: %.3f ms   e2e wall: %.3f ms\n", e2e_kernel, e2e_wall);
    std::printf("  vs NEON 1t: kernel %.2fx, e2e %.2fx\n", neon_1t_ms / e2e_kernel,
                neon_1t_ms / e2e_wall);
    std::printf("  vs NEON MT: kernel %.2fx, e2e %.2fx\n", neon_mt_ms / e2e_kernel,
                neon_mt_ms / e2e_wall);
    if (e2e_wall > neon_mt_ms) {
        std::printf("  => For host Mat, prefer NEON. Use GPU when data already on GPU.\n");
    }

    std::printf("\n%d x %d  traffic ~= %.1f MB (read+write)\n", args.width, args.height,
                bytes / (1024.0 * 1024.0));

    clReleaseProgram(prog_f);
    clReleaseProgram(prog_f1);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return all_ok ? 0 : 1;
}
