// Standalone enhanceBrightness microbench (no findmax).
// Compare: 2D 1px | 2D half4 (vload4) | 1D grid-stride half4
// max_value is float by value. Default 5760x4320.
//
// Usage:
//   ./ocl_test_enhance_only [--width W] [--height H] [--runs 30]
//                           [--lwsx 16] [--lwsy 16] [--lws-opt 256] [--nwg 256]
//                           [--max 0.25]

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
    std::string path_2d = "enhance_brightness.cl";
    std::string path_2dv4 = "enhance_brightness_2d_v4.cl";
    std::string path_1d = "enhance_brightness_opt.cl";
    int width = 5760;
    int height = 4320;
    int runs = 30;
    int warmup = 5;
    int lwsx = 16;
    int lwsy = 16;
    int lws_opt = 256;
    int nwg = 256;
    float max_value = 0.25f;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto take = [&](const char* key, std::string& dst) {
            if (std::strcmp(argv[i], key) == 0 && i + 1 < argc) {
                dst = argv[++i];
                return true;
            }
            return false;
        };
        if (take("--2d", a.path_2d) || take("--2dv4", a.path_2dv4) || take("--1d", a.path_1d)) {
            continue;
        }
        if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            a.width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            a.height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            a.runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            a.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lwsx") == 0 && i + 1 < argc) {
            a.lwsx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lwsy") == 0 && i + 1 < argc) {
            a.lwsy = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lws-opt") == 0 && i + 1 < argc) {
            a.lws_opt = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nwg") == 0 && i + 1 < argc) {
            a.nwg = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            a.max_value = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--width W] [--height H] [--max 0.25] [--runs N]\n"
                        "          [--lwsx 16] [--lwsy 16] [--lws-opt 256] [--nwg 256]\n",
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
        return static_cast<uint16_t>(sign | (mant >> (1 - exp + 13)));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
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

void fillHalfImage(std::vector<uint16_t>& img, int w, int h, float peak) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    img.resize(n);
    const size_t peak_i = n > 0 ? n - 1 : 0;
    for (size_t i = 0; i < n; ++i) {
        const int x = static_cast<int>(i % static_cast<size_t>(w));
        const int y = static_cast<int>(i / static_cast<size_t>(w));
        float v = static_cast<float>(((x * 13 + y * 7) % 1000)) / 1000.f;
        if (v >= peak) {
            v = peak * 0.5f;
        }
        if (i == peak_i) {
            v = peak;
        }
        img[i] = floatToHalf(v);
    }
}

void cpuEnhance(std::vector<uint16_t>& img, float max_value) {
    const float div = std::min(1.0f, max_value);
    const float inv = (div > 1e-7f) ? (1.0f / div) : 1.0f;
    for (uint16_t& h : img) {
        h = floatToHalf(halfToFloat(h) * inv);
    }
}

float maxAbsDiffHalf(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(halfToFloat(a[i]) - halfToFloat(b[i])));
    }
    return m;
}

cl_program buildProgram(cl_context ctx, cl_device_id device, const std::string& path,
                        const char* opts) {
    const std::string src = readText(path);
    const char* srcp = src.c_str();
    size_t srcl = src.size();
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, &srcl, &err);
    OCL_CHECK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &device, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::fprintf(stderr, "Build failed (%s):\n%s\n", path.c_str(), log.data());
        std::exit(1);
    }
    return prog;
}

void setEnhance(cl_kernel kn, cl_mem src, unsigned w, unsigned h, float max_value) {
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &src), "a0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(unsigned), &w), "a1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(unsigned), &h), "a2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(float), &max_value), "a3");
}

struct Case {
    const char* name;
    cl_kernel kn = nullptr;
    int dim = 1;
    size_t gws[2] = {0, 0};
    size_t lws[2] = {0, 0};
};

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    const Args args = parseArgs(argc, argv);
    if (args.lws_opt <= 0 || (args.lws_opt & (args.lws_opt - 1)) != 0 || args.nwg <= 0) {
        std::fprintf(stderr, "bad --lws-opt / --nwg\n");
        return 1;
    }
    if (!(args.max_value > 0.f && args.max_value < 1.f)) {
        std::fprintf(stderr, "--max should be in (0,1) so enhance is not a no-op\n");
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
    const float mv = args.max_value;

    std::vector<uint16_t> host_in;
    fillHalfImage(host_in, W, H, mv);
    std::vector<uint16_t> host_ref = host_in;
    cpuEnhance(host_ref, mv);

    cl_uint np = 0;
    OCL_CHECK(clGetPlatformIDs(0, nullptr, &np), "platforms");
    std::vector<cl_platform_id> platforms(np);
    OCL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr), "platforms");
    cl_device_id device = pickDevice(platforms[0]);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    OCL_CHECK(err, "ctx");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    OCL_CHECK(err, "queue");

    char opts_1d[128];
    std::snprintf(opts_1d, sizeof(opts_1d), "-cl-std=CL1.2 -DLSIZE=%d", args.lws_opt);

    cl_program p2d = buildProgram(ctx, device, args.path_2d, "-cl-std=CL1.2");
    cl_program p2v = buildProgram(ctx, device, args.path_2dv4, "-cl-std=CL1.2");
    cl_program p1d = buildProgram(ctx, device, args.path_1d, opts_1d);

    Case c2d, c2v, c1d;
    c2d.name = "2d_1px";
    c2d.kn = clCreateKernel(p2d, "enhanceBrightness", &err);
    OCL_CHECK(err, "2d kn");
    c2d.dim = 2;
    c2d.lws[0] = static_cast<size_t>(args.lwsx);
    c2d.lws[1] = static_cast<size_t>(args.lwsy);
    c2d.gws[0] = ((static_cast<size_t>(W) + c2d.lws[0] - 1) / c2d.lws[0]) * c2d.lws[0];
    c2d.gws[1] = ((static_cast<size_t>(H) + c2d.lws[1] - 1) / c2d.lws[1]) * c2d.lws[1];

    c2v.name = "2d_v4";
    c2v.kn = clCreateKernel(p2v, "enhanceBrightness", &err);
    OCL_CHECK(err, "2dv4 kn");
    c2v.dim = 2;
    c2v.lws[0] = static_cast<size_t>(args.lwsx);
    c2v.lws[1] = static_cast<size_t>(args.lwsy);
    const size_t groups_x = (static_cast<size_t>(W) + 3) / 4;
    c2v.gws[0] = ((groups_x + c2v.lws[0] - 1) / c2v.lws[0]) * c2v.lws[0];
    c2v.gws[1] = ((static_cast<size_t>(H) + c2v.lws[1] - 1) / c2v.lws[1]) * c2v.lws[1];

    c1d.name = "1d_stride";
    c1d.kn = clCreateKernel(p1d, "enhanceBrightness", &err);
    OCL_CHECK(err, "1d kn");
    c1d.dim = 1;
    c1d.lws[0] = static_cast<size_t>(args.lws_opt);
    c1d.gws[0] = c1d.lws[0] * static_cast<size_t>(args.nwg);

    cl_mem buf_gold =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_in.data(), &err);
    OCL_CHECK(err, "gold");
    cl_mem buf_src = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    OCL_CHECK(err, "src");

    const unsigned wu = static_cast<unsigned>(W);
    const unsigned hu = static_cast<unsigned>(H);
    setEnhance(c2d.kn, buf_src, wu, hu, mv);
    setEnhance(c2v.kn, buf_src, wu, hu, mv);
    setEnhance(c1d.kn, buf_src, wu, hu, mv);

    auto restore = [&] {
        OCL_CHECK(clEnqueueCopyBuffer(queue, buf_gold, buf_src, 0, 0, bytes, 0, nullptr, nullptr),
                  "restore");
        OCL_CHECK(clFinish(queue), "restore fin");
    };

    std::vector<uint16_t> out(nPix);
    auto check = [&](const Case& c) {
        restore();
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, c.kn, c.dim, nullptr, c.gws, c.lws, 0, nullptr, &ev),
                  c.name);
        OCL_CHECK(clWaitForEvents(1, &ev), "wait");
        clReleaseEvent(ev);
        OCL_CHECK(clEnqueueReadBuffer(queue, buf_src, CL_TRUE, 0, bytes, out.data(), 0, nullptr,
                                      nullptr),
                  "read");
        const float errm = maxAbsDiffHalf(out, host_ref);
        const float vs_in = maxAbsDiffHalf(out, host_in);
        const bool ok = errm < 2e-2f && vs_in > 1e-3f;
        std::printf("[%s] vs_cpu=%.3e vs_in=%.3e %s\n", c.name, errm, vs_in, ok ? "OK" : "FAIL");
        if (!ok) {
            std::exit(1);
        }
    };

    auto bench = [&](const Case& c) -> std::pair<double, double> {
        for (int i = 0; i < args.warmup; ++i) {
            restore();
            cl_event ev = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, c.kn, c.dim, nullptr, c.gws, c.lws, 0, nullptr,
                                             &ev),
                      "warm");
            OCL_CHECK(clWaitForEvents(1, &ev), "ww");
            clReleaseEvent(ev);
        }
        double sum = 0, best = 1e300;
        for (int i = 0; i < args.runs; ++i) {
            restore();
            OCL_CHECK(clFinish(queue), "fin");
            cl_event ev = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, c.kn, c.dim, nullptr, c.gws, c.lws, 0, nullptr,
                                             &ev),
                      "t");
            const double ms = profileMs(ev);
            clReleaseEvent(ev);
            sum += ms;
            best = std::min(best, ms);
        }
        return {sum / args.runs, best};
    };

    std::printf("############################################################\n");
    std::printf("# enhanceBrightness ONLY (no findmax)\n");
    std::printf("# max_value=%.4f (float by value)  size=%dx%d\n", mv, W, H);
    std::printf("############################################################\n");
    std::printf("device: %s\n", deviceName(device).c_str());
    std::printf("2d_1px:   gws=(%zu,%zu) lws=(%zu,%zu)\n", c2d.gws[0], c2d.gws[1], c2d.lws[0],
                c2d.lws[1]);
    std::printf("2d_v4:    gws=(%zu,%zu) lws=(%zu,%zu)  (~4 px/WI in X)\n", c2v.gws[0], c2v.gws[1],
                c2v.lws[0], c2v.lws[1]);
    std::printf("1d_stride:gws=%zu lws=%zu nwg=%d\n\n", c1d.gws[0], c1d.lws[0], args.nwg);

    check(c2d);
    check(c2v);
    check(c1d);

    const auto [a_avg, a_best] = bench(c2d);
    const auto [b_avg, b_best] = bench(c2v);
    const auto [c_avg, c_best] = bench(c1d);

    std::printf("\n--- enhance kernel time only (avg %d) ---\n", args.runs);
    std::printf("  %-12s %10s %10s %10s\n", "impl", "avg_ms", "best_ms", "vs_2d_1px");
    std::printf("  %-12s %10.3f %10.3f %9.2fx\n", "2d_1px", a_avg, a_best, 1.0);
    std::printf("  %-12s %10.3f %10.3f %9.2fx\n", "2d_v4", b_avg, b_best,
                a_avg / std::max(b_avg, 1e-9));
    std::printf("  %-12s %10.3f %10.3f %9.2fx\n", "1d_stride", c_avg, c_best,
                a_avg / std::max(c_avg, 1e-9));

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_gold);
    clReleaseKernel(c2d.kn);
    clReleaseKernel(c2v.kn);
    clReleaseKernel(c1d.kn);
    clReleaseProgram(p2d);
    clReleaseProgram(p2v);
    clReleaseProgram(p1d);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
