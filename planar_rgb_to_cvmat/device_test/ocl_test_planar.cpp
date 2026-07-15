// HarmonyOS OpenCL: planar RGB -> CV_32FC3 (interleaved BGR float).
// Builds CPU reference, runs GPU kernel, checks max abs error, profiles kernel time.
//
// Usage:
//   ./ocl_test_planar [--kernel PATH] [--width W] [--height H] [--runs N]

#include <CL/cl.h>
#ifdef OCR_OPENCL_DLOPEN
#include "opencl_dynload.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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

void cpuRefBgr(const float* r, const float* g, const float* b, float* dst, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t si = static_cast<size_t>(y) * w + x;
            float* out = dst + si * 3;
            out[0] = b[si];
            out[1] = g[si];
            out[2] = r[si];
        }
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

// Average wall time of CPU planar -> interleaved BGR conversion.
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
    const char* name;
    float max_diff;
    double gpu_ms;
    double cpu_ms;
    bool ok;
};

CaseResult runFloatSeparate(cl_context ctx, cl_command_queue q, cl_program prog, const Args& args,
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

    cl_kernel kn = clCreateKernel(prog, "planar_rgb_to_cv32fc3", &err);
    OCL_CHECK(err, "kernel");
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_r), "arg0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &buf_g), "arg1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(cl_mem), &buf_b), "arg2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &buf_out), "arg3");
    OCL_CHECK(clSetKernelArg(kn, 4, sizeof(int), &w), "arg4");
    OCL_CHECK(clSetKernelArg(kn, 5, sizeof(int), &h), "arg5");
    OCL_CHECK(clSetKernelArg(kn, 6, sizeof(int), &src_stride), "arg6");
    OCL_CHECK(clSetKernelArg(kn, 7, sizeof(int), &dst_stride), "arg7");

    size_t global[2] = {static_cast<size_t>(w), static_cast<size_t>(h)};
    for (int i = 0; i < args.warmup; ++i) {
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                  "warmup");
    }
    OCL_CHECK(clFinish(q), "finish warmup");

    double sum_ms = 0.0;
    for (int i = 0; i < args.runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, nullptr, 0, nullptr, &ev),
                  "enqueue");
        sum_ms += profileMs(ev);
        clReleaseEvent(ev);
    }

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

    CaseResult res{};
    res.name = "float separate planes";
    res.max_diff = diff;
    res.gpu_ms = sum_ms / args.runs;
    res.cpu_ms = 0.0;
    res.ok = diff <= 1e-5f;
    return res;
}

CaseResult runFloatPacked(cl_context ctx, cl_command_queue q, cl_program prog, const Args& args,
                          const std::vector<float>& r, const std::vector<float>& g,
                          const std::vector<float>& b, const std::vector<float>& ref) {
    const int w = args.width;
    const int h = args.height;
    const size_t n = static_cast<size_t>(w) * h;
    const size_t out_n = n * 3;
    const int src_stride = w;
    const int dst_stride = w * 3;

    std::vector<float> packed(n * 3);
    std::memcpy(packed.data(), r.data(), n * sizeof(float));
    std::memcpy(packed.data() + n, g.data(), n * sizeof(float));
    std::memcpy(packed.data() + 2 * n, b.data(), n * sizeof(float));

    cl_int err = 0;
    cl_mem buf_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   packed.size() * sizeof(float), packed.data(), &err);
    OCL_CHECK(err, "buf_in");
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "buf_out");

    cl_kernel kn = clCreateKernel(prog, "planar_rgb_packed_to_cv32fc3", &err);
    OCL_CHECK(err, "kernel packed");
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_in), "arg0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &buf_out), "arg1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(int), &w), "arg2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(int), &h), "arg3");
    OCL_CHECK(clSetKernelArg(kn, 4, sizeof(int), &src_stride), "arg4");
    OCL_CHECK(clSetKernelArg(kn, 5, sizeof(int), &dst_stride), "arg5");

    size_t global[2] = {static_cast<size_t>(w), static_cast<size_t>(h)};
    for (int i = 0; i < args.warmup; ++i) {
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                  "warmup");
    }
    OCL_CHECK(clFinish(q), "finish warmup");

    double sum_ms = 0.0;
    for (int i = 0; i < args.runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, nullptr, 0, nullptr, &ev),
                  "enqueue");
        sum_ms += profileMs(ev);
        clReleaseEvent(ev);
    }

    std::vector<float> out(out_n);
    OCL_CHECK(clEnqueueReadBuffer(q, buf_out, CL_TRUE, 0, out_n * sizeof(float), out.data(), 0,
                                  nullptr, nullptr),
              "read");
    const float diff = maxAbsDiff(out.data(), ref.data(), out_n);

    clReleaseKernel(kn);
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);

    CaseResult res{};
    res.name = "float packed [R|G|B]";
    res.max_diff = diff;
    res.gpu_ms = sum_ms / args.runs;
    res.cpu_ms = 0.0;
    res.ok = diff <= 1e-5f;
    return res;
}

CaseResult runUcharSeparate(cl_context ctx, cl_command_queue q, cl_program prog, const Args& args) {
    const int w = args.width;
    const int h = args.height;
    const size_t n = static_cast<size_t>(w) * h;
    const size_t out_n = n * 3;
    const int src_stride = w;
    const int dst_stride = w * 3;

    std::vector<unsigned char> r, g, b;
    fillUcharPlanes(r, g, b, w, h);
    std::vector<float> ref(out_n);
    cpuRefBgrUchar(r.data(), g.data(), b.data(), ref.data(), w, h);

    cl_int err = 0;
    cl_mem buf_r =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, r.data(), &err);
    OCL_CHECK(err, "buf_r u8");
    cl_mem buf_g =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, g.data(), &err);
    OCL_CHECK(err, "buf_g u8");
    cl_mem buf_b =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, b.data(), &err);
    OCL_CHECK(err, "buf_b u8");
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_n * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "buf_out u8");

    cl_kernel kn = clCreateKernel(prog, "planar_rgb_to_cv32fc3", &err);
    OCL_CHECK(err, "kernel u8");
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_r), "arg0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &buf_g), "arg1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(cl_mem), &buf_b), "arg2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &buf_out), "arg3");
    OCL_CHECK(clSetKernelArg(kn, 4, sizeof(int), &w), "arg4");
    OCL_CHECK(clSetKernelArg(kn, 5, sizeof(int), &h), "arg5");
    OCL_CHECK(clSetKernelArg(kn, 6, sizeof(int), &src_stride), "arg6");
    OCL_CHECK(clSetKernelArg(kn, 7, sizeof(int), &dst_stride), "arg7");

    size_t global[2] = {static_cast<size_t>(w), static_cast<size_t>(h)};
    for (int i = 0; i < args.warmup; ++i) {
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                  "warmup");
    }
    OCL_CHECK(clFinish(q), "finish warmup");

    double sum_ms = 0.0;
    for (int i = 0; i < args.runs; ++i) {
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(q, kn, 2, nullptr, global, nullptr, 0, nullptr, &ev),
                  "enqueue");
        sum_ms += profileMs(ev);
        clReleaseEvent(ev);
    }

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

    CaseResult res{};
    res.name = "uchar separate (/255)";
    res.max_diff = diff;
    res.gpu_ms = sum_ms / args.runs;
    res.cpu_ms = 0.0;
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
    cl_command_queue q =
        clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    OCL_CHECK(err, "queue");

    cl_program prog_f = buildProgram(ctx, dev, src, nullptr);
    cl_program prog_u8 = buildProgram(ctx, dev, src, "-DINPUT_UCHAR");

    std::vector<float> r, g, b;
    fillPlanes(r, g, b, args.width, args.height);
    std::vector<float> ref(static_cast<size_t>(args.width) * args.height * 3);
    cpuRefBgr(r.data(), g.data(), b.data(), ref.data(), args.width, args.height);

    std::vector<float> cpu_dst(ref.size());
    const double cpu_float_ms = benchCpuMs(args.warmup, args.runs, [&] {
        cpuRefBgr(r.data(), g.data(), b.data(), cpu_dst.data(), args.width, args.height);
    });

    std::vector<unsigned char> ur, ug, ub;
    fillUcharPlanes(ur, ug, ub, args.width, args.height);
    const double cpu_uchar_ms = benchCpuMs(args.warmup, args.runs, [&] {
        cpuRefBgrUchar(ur.data(), ug.data(), ub.data(), cpu_dst.data(), args.width, args.height);
    });

    CaseResult c0 = runFloatSeparate(ctx, q, prog_f, args, r, g, b, ref);
    CaseResult c1 = runFloatPacked(ctx, q, prog_f, args, r, g, b, ref);
    CaseResult c2 = runUcharSeparate(ctx, q, prog_u8, args);
    c0.cpu_ms = cpu_float_ms;
    c1.cpu_ms = cpu_float_ms;
    c2.cpu_ms = cpu_uchar_ms;

    std::printf("=== results (CPU convert vs GPU kernel, OpenCV BGR float) ===\n");
    std::printf("  (cpu = host planar->CV_32FC3 convert; gpu = OpenCL kernel only, no H2D/D2H)\n");
    bool all_ok = true;
    for (const CaseResult& c : {c0, c1, c2}) {
        const double speedup = c.gpu_ms > 0.0 ? (c.cpu_ms / c.gpu_ms) : 0.0;
        std::printf("  %-28s  max_diff=%.3e  cpu=%.3f ms  gpu=%.3f ms  (%.2fx)  %s\n", c.name,
                    c.max_diff, c.cpu_ms, c.gpu_ms, speedup, c.ok ? "OK" : "FAIL");
        all_ok = all_ok && c.ok;
    }
    std::printf("\n%d x %d x 3 floats = %.1f MB output\n", args.width, args.height,
                args.width * args.height * 3 * 4 / (1024.0 * 1024.0));

    clReleaseProgram(prog_f);
    clReleaseProgram(prog_u8);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return all_ok ? 0 : 1;
}
