// Standalone enhanceBrightness microbench (no findmax).
// Goal: see if we can beat current 1d half4 (bandwidth-bound scale).
//
// Cases:
//   1d_h4     - current opt (max_value float, half4, float convert)
//   1d_h8_inv - half8 * half(inv)
//   1d_h16_inv- half16 * half(inv)
//   + nwg sweep on best candidate
//
// Usage:
//   ./ocl_test_enhance_only [--width W] [--height H] [--runs 30]
//                           [--lws-opt 256] [--nwg 256] [--max 0.25]

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
    std::string path_h4 = "enhance_brightness_opt.cl";
    std::string path_h8 = "enhance_brightness_opt_h8.cl";
    std::string path_h16 = "enhance_brightness_opt_h16.cl";
    int width = 5760;
    int height = 4320;
    int runs = 30;
    int warmup = 5;
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
        if (take("--h4", a.path_h4) || take("--h8", a.path_h8) || take("--h16", a.path_h16)) {
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
        } else if (std::strcmp(argv[i], "--lws-opt") == 0 && i + 1 < argc) {
            a.lws_opt = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nwg") == 0 && i + 1 < argc) {
            a.nwg = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            a.max_value = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--width W] [--height H] [--max 0.25] [--runs N]\n"
                        "          [--lws-opt 256] [--nwg 256]\n",
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

void setArgs(cl_kernel kn, cl_mem src, unsigned w, unsigned h, float arg3) {
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &src), "a0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(unsigned), &w), "a1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(unsigned), &h), "a2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(float), &arg3), "a3");
}

struct Case {
    const char* name = nullptr;
    cl_kernel kn = nullptr;
    size_t gws = 0;
    size_t lws = 0;
    float arg3 = 0.f;  // max_value or inv
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
    const float inv = 1.0f / std::min(1.0f, mv);

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

    char opts[128];
    std::snprintf(opts, sizeof(opts), "-cl-std=CL1.2 -DLSIZE=%d", args.lws_opt);

    cl_program p4 = buildProgram(ctx, device, args.path_h4, opts);
    cl_program p8 = buildProgram(ctx, device, args.path_h8, opts);
    cl_program p16 = buildProgram(ctx, device, args.path_h16, opts);

    const size_t lws = static_cast<size_t>(args.lws_opt);
    const size_t gws = lws * static_cast<size_t>(args.nwg);

    Case cases[3];
    cases[0].name = "1d_h4";
    cases[0].kn = clCreateKernel(p4, "enhanceBrightness", &err);
    OCL_CHECK(err, "h4");
    cases[0].gws = gws;
    cases[0].lws = lws;
    cases[0].arg3 = mv;  // existing API: max_value

    cases[1].name = "1d_h8_inv";
    cases[1].kn = clCreateKernel(p8, "enhanceBrightness", &err);
    OCL_CHECK(err, "h8");
    cases[1].gws = gws;
    cases[1].lws = lws;
    cases[1].arg3 = inv;

    cases[2].name = "1d_h16_inv";
    cases[2].kn = clCreateKernel(p16, "enhanceBrightness", &err);
    OCL_CHECK(err, "h16");
    cases[2].gws = gws;
    cases[2].lws = lws;
    cases[2].arg3 = inv;

    cl_mem buf_gold =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_in.data(), &err);
    OCL_CHECK(err, "gold");
    cl_mem buf_src = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    OCL_CHECK(err, "src");

    const unsigned wu = static_cast<unsigned>(W);
    const unsigned hu = static_cast<unsigned>(H);
    for (Case& c : cases) {
        setArgs(c.kn, buf_src, wu, hu, c.arg3);
    }

    auto restore = [&] {
        OCL_CHECK(clEnqueueCopyBuffer(queue, buf_gold, buf_src, 0, 0, bytes, 0, nullptr, nullptr),
                  "restore");
        OCL_CHECK(clFinish(queue), "restore fin");
    };

    std::vector<uint16_t> out(nPix);
    auto check = [&](Case& c) {
        setArgs(c.kn, buf_src, wu, hu, c.arg3);
        restore();
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, c.kn, 1, nullptr, &c.gws, &c.lws, 0, nullptr, &ev),
                  c.name);
        OCL_CHECK(clWaitForEvents(1, &ev), "wait");
        clReleaseEvent(ev);
        OCL_CHECK(clEnqueueReadBuffer(queue, buf_src, CL_TRUE, 0, bytes, out.data(), 0, nullptr,
                                      nullptr),
                  "read");
        const float errm = maxAbsDiffHalf(out, host_ref);
        const float vs_in = maxAbsDiffHalf(out, host_in);
        // half*half may be slightly looser than float path
        const bool ok = errm < 5e-2f && vs_in > 1e-3f;
        std::printf("[%s] vs_cpu=%.3e vs_in=%.3e %s\n", c.name, errm, vs_in, ok ? "OK" : "FAIL");
        if (!ok) {
            std::exit(1);
        }
    };

    auto bench = [&](Case& c) -> std::pair<double, double> {
        setArgs(c.kn, buf_src, wu, hu, c.arg3);
        for (int i = 0; i < args.warmup; ++i) {
            restore();
            cl_event ev = nullptr;
            OCL_CHECK(
                clEnqueueNDRangeKernel(queue, c.kn, 1, nullptr, &c.gws, &c.lws, 0, nullptr, &ev),
                "warm");
            OCL_CHECK(clWaitForEvents(1, &ev), "ww");
            clReleaseEvent(ev);
        }
        double sum = 0, best = 1e300;
        for (int i = 0; i < args.runs; ++i) {
            restore();
            OCL_CHECK(clFinish(queue), "fin");
            cl_event ev = nullptr;
            OCL_CHECK(
                clEnqueueNDRangeKernel(queue, c.kn, 1, nullptr, &c.gws, &c.lws, 0, nullptr, &ev),
                "t");
            const double ms = profileMs(ev);
            clReleaseEvent(ev);
            sum += ms;
            best = std::min(best, ms);
        }
        return {sum / args.runs, best};
    };

    const double gb_rw = (2.0 * static_cast<double>(bytes)) / 1e9;  // read+write

    std::printf("############################################################\n");
    std::printf("# enhance ONLY - push past half4?\n");
    std::printf("# size=%dx%d  max=%.4f inv=%.4f  traffic=%.2f GB (R+W)\n", W, H, mv, inv, gb_rw);
    std::printf("# If GB/s plateaus across impls => memory-bound, little left.\n");
    std::printf("############################################################\n");
    std::printf("device: %s\n", deviceName(device).c_str());
    std::printf("launch: gws=%zu lws=%zu nwg=%d\n\n", gws, lws, args.nwg);

    for (Case& c : cases) {
        check(c);
    }

    double avgs[3], bests[3];
    for (int i = 0; i < 3; ++i) {
        const auto r = bench(cases[i]);
        avgs[i] = r.first;
        bests[i] = r.second;
    }

    std::printf("\n--- results (avg %d) ---\n", args.runs);
    std::printf("  %-12s %10s %10s %10s %10s\n", "impl", "avg_ms", "best_ms", "vs_h4", "GB/s");
    double best_avg = avgs[0];
    int best_i = 0;
    for (int i = 0; i < 3; ++i) {
        const double gbs = gb_rw / (avgs[i] / 1e3);
        std::printf("  %-12s %10.3f %10.3f %9.2fx %10.1f\n", cases[i].name, avgs[i], bests[i],
                    avgs[0] / std::max(avgs[i], 1e-9), gbs);
        if (avgs[i] < best_avg) {
            best_avg = avgs[i];
            best_i = i;
        }
    }

    // nwg sweep on winner
    static const int kNwgs[] = {32, 64, 128, 256, 512, 1024};
    std::printf("\n--- nwg sweep on %s (lws=%zu) ---\n", cases[best_i].name, lws);
    std::printf("  %6s %10s %10s\n", "nwg", "avg_ms", "GB/s");
    double sweep_best = 1e300;
    int sweep_nwg = args.nwg;
    for (int nwg : kNwgs) {
        cases[best_i].gws = lws * static_cast<size_t>(nwg);
        const auto [avg, best] = bench(cases[best_i]);
        (void)best;
        const double gbs = gb_rw / (avg / 1e3);
        std::printf("  %6d %10.3f %10.1f\n", nwg, avg, gbs);
        if (avg < sweep_best) {
            sweep_best = avg;
            sweep_nwg = nwg;
        }
    }
    std::printf("\nbest: %s @ nwg=%d  avg=%.3f ms  (%.1f GB/s R+W)\n", cases[best_i].name, sweep_nwg,
                sweep_best, gb_rw / (sweep_best / 1e3));
    std::printf("note: if all ~same GB/s, kernel is memory-bound; next wins are skip-if-max>=1\n"
                "      or fuse with a later consumer (avoid extra write).\n");

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_gold);
    for (Case& c : cases) {
        clReleaseKernel(c.kn);
    }
    clReleaseProgram(p4);
    clReleaseProgram(p8);
    clReleaseProgram(p16);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
