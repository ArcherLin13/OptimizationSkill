// Standalone f32 -> f16 GPU microbench (HarmonyOS OpenCL).
// Compare:
//   A) 1d_n   gws ~= n, 1 float/WI
//   B) 2d     gws ~= (W,H), 1 float/WI
//   C) stride fixed gws = lws*nwg, float4 grid-stride
//
// Usage:
//   ./ocl_test_f32_to_f16 [--width 5760] [--height 4320] [--runs 30]
//                         [--lws1d 256] [--lwsx 16] [--lwsy 16]
//                         [--lws-opt 256] [--nwg 256]

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
    std::string path_1d = "f32_to_f16_1d_n.cl";
    std::string path_2d = "f32_to_f16_2d.cl";
    std::string path_stride = "f32_to_f16_stride.cl";
    int width = 5760;
    int height = 4320;
    int runs = 30;
    int warmup = 5;
    int lws1d = 256;
    int lwsx = 16;
    int lwsy = 16;
    int lws_opt = 256;
    int nwg = 256;
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
        if (take("--1d", a.path_1d) || take("--2d", a.path_2d) || take("--stride", a.path_stride)) {
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
        } else if (std::strcmp(argv[i], "--lws1d") == 0 && i + 1 < argc) {
            a.lws1d = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lwsx") == 0 && i + 1 < argc) {
            a.lwsx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lwsy") == 0 && i + 1 < argc) {
            a.lwsy = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lws-opt") == 0 && i + 1 < argc) {
            a.lws_opt = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nwg") == 0 && i + 1 < argc) {
            a.nwg = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--width W] [--height H] [--runs N]\n"
                        "          [--lws1d 256] [--lwsx 16] [--lwsy 16]\n"
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

void fillFloatBuffer(std::vector<float>& src, int w, int h) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    src.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const int x = static_cast<int>(i % static_cast<size_t>(w));
        const int y = static_cast<int>(i / static_cast<size_t>(w));
        src[i] = static_cast<float>(((x * 13 + y * 7) % 1000)) / 500.f - 1.f;
    }
}

void cpuF32ToF16(const std::vector<float>& src, std::vector<uint16_t>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = floatToHalf(src[i]);
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

struct Case {
    const char* name = nullptr;
    cl_kernel kn = nullptr;
    int dim = 1;
    size_t gws[2] = {0, 0};
    size_t lws[2] = {0, 0};
    bool use_2d_args = false;  // width+height vs n
};

void setArgs1d(cl_kernel kn, cl_mem src, cl_mem dst, unsigned n) {
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &src), "a0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &dst), "a1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(unsigned), &n), "a2");
}

void setArgs2d(cl_kernel kn, cl_mem src, cl_mem dst, unsigned w, unsigned h) {
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &src), "a0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(cl_mem), &dst), "a1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(unsigned), &w), "a2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(unsigned), &h), "a3");
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    const Args args = parseArgs(argc, argv);
    if (args.width <= 0 || args.height <= 0) {
        std::fprintf(stderr, "--width/--height must be > 0\n");
        return 1;
    }
    if (args.lws_opt <= 0 || (args.lws_opt & (args.lws_opt - 1)) != 0 || args.nwg <= 0) {
        std::fprintf(stderr, "bad --lws-opt / --nwg\n");
        return 1;
    }
    {
        const unsigned long long n64 =
            static_cast<unsigned long long>(args.width) * static_cast<unsigned long long>(args.height);
        if (n64 > 0xffffffffULL) {
            std::fprintf(stderr, "width*height overflows uint\n");
            return 1;
        }
    }

#ifdef OCR_OPENCL_DLOPEN
    if (!opencl_load()) {
        return 1;
    }
#endif

    const int W = args.width;
    const int H = args.height;
    const unsigned n = static_cast<unsigned>(W) * static_cast<unsigned>(H);
    const size_t src_bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t dst_bytes = static_cast<size_t>(n) * sizeof(uint16_t);

    std::vector<float> host_src;
    fillFloatBuffer(host_src, W, H);
    std::vector<uint16_t> host_ref;
    cpuF32ToF16(host_src, host_ref);

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

    char opts_stride[128];
    std::snprintf(opts_stride, sizeof(opts_stride), "-cl-std=CL1.2 -DLSIZE=%d", args.lws_opt);

    cl_program p1d = buildProgram(ctx, device, args.path_1d, "-cl-std=CL1.2");
    cl_program p2d = buildProgram(ctx, device, args.path_2d, "-cl-std=CL1.2");
    cl_program p_stride = buildProgram(ctx, device, args.path_stride, opts_stride);

    Case cases[3];
    cases[0].name = "1d_n";
    cases[0].kn = clCreateKernel(p1d, "f32_to_f16", &err);
    OCL_CHECK(err, "1d kn");
    cases[0].dim = 1;
    cases[0].lws[0] = static_cast<size_t>(args.lws1d);
    cases[0].gws[0] = ((static_cast<size_t>(n) + cases[0].lws[0] - 1) / cases[0].lws[0]) *
                       cases[0].lws[0];
    cases[0].use_2d_args = false;

    cases[1].name = "2d_wh";
    cases[1].kn = clCreateKernel(p2d, "f32_to_f16", &err);
    OCL_CHECK(err, "2d kn");
    cases[1].dim = 2;
    cases[1].lws[0] = static_cast<size_t>(args.lwsx);
    cases[1].lws[1] = static_cast<size_t>(args.lwsy);
    cases[1].gws[0] = ((static_cast<size_t>(W) + cases[1].lws[0] - 1) / cases[1].lws[0]) *
                       cases[1].lws[0];
    cases[1].gws[1] = ((static_cast<size_t>(H) + cases[1].lws[1] - 1) / cases[1].lws[1]) *
                       cases[1].lws[1];
    cases[1].use_2d_args = true;

    cases[2].name = "stride_f4";
    cases[2].kn = clCreateKernel(p_stride, "f32_to_f16", &err);
    OCL_CHECK(err, "stride kn");
    cases[2].dim = 1;
    cases[2].lws[0] = static_cast<size_t>(args.lws_opt);
    cases[2].gws[0] = cases[2].lws[0] * static_cast<size_t>(args.nwg);
    cases[2].use_2d_args = false;

    cl_mem buf_src =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src_bytes, host_src.data(), &err);
    OCL_CHECK(err, "src");
    cl_mem buf_dst = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, dst_bytes, nullptr, &err);
    OCL_CHECK(err, "dst");

    const unsigned wu = static_cast<unsigned>(W);
    const unsigned hu = static_cast<unsigned>(H);

    auto setCaseArgs = [&](Case& c) {
        if (c.use_2d_args) {
            setArgs2d(c.kn, buf_src, buf_dst, wu, hu);
        } else {
            setArgs1d(c.kn, buf_src, buf_dst, n);
        }
    };
    for (Case& c : cases) {
        setCaseArgs(c);
    }

    std::vector<uint16_t> out(n);
    auto check = [&](Case& c) {
        setCaseArgs(c);
        cl_event ev = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, c.kn, c.dim, nullptr, c.gws, c.lws, 0, nullptr, &ev),
                  c.name);
        OCL_CHECK(clWaitForEvents(1, &ev), "wait");
        clReleaseEvent(ev);
        OCL_CHECK(clEnqueueReadBuffer(queue, buf_dst, CL_TRUE, 0, dst_bytes, out.data(), 0, nullptr,
                                      nullptr),
                  "read");
        const float errm = maxAbsDiffHalf(out, host_ref);
        const bool ok = errm < 5e-2f;
        std::printf("[%s] max_abs_diff=%.3e %s\n", c.name, errm, ok ? "OK" : "FAIL");
        if (!ok) {
            std::exit(1);
        }
    };

    auto bench = [&](Case& c) -> std::pair<double, double> {
        setCaseArgs(c);
        for (int i = 0; i < args.warmup; ++i) {
            cl_event ev = nullptr;
            OCL_CHECK(
                clEnqueueNDRangeKernel(queue, c.kn, c.dim, nullptr, c.gws, c.lws, 0, nullptr, &ev),
                "warm");
            OCL_CHECK(clWaitForEvents(1, &ev), "ww");
            clReleaseEvent(ev);
        }
        double sum = 0, best = 1e300;
        for (int i = 0; i < args.runs; ++i) {
            OCL_CHECK(clFinish(queue), "fin");
            cl_event ev = nullptr;
            OCL_CHECK(
                clEnqueueNDRangeKernel(queue, c.kn, c.dim, nullptr, c.gws, c.lws, 0, nullptr, &ev),
                "t");
            const double ms = profileMs(ev);
            clReleaseEvent(ev);
            sum += ms;
            best = std::min(best, ms);
        }
        return {sum / args.runs, best};
    };

    const double gb_rw = (static_cast<double>(src_bytes) + static_cast<double>(dst_bytes)) / 1e9;

    std::printf("############################################################\n");
    std::printf("# f32 -> f16 GPU microbench\n");
    std::printf("# size=%dx%d  n=%u  traffic=%.2f GB (read+write)\n", W, H, n, gb_rw);
    std::printf("############################################################\n");
    std::printf("device: %s\n", deviceName(device).c_str());
    std::printf("1d_n:      gws=%zu lws=%zu\n", cases[0].gws[0], cases[0].lws[0]);
    std::printf("2d_wh:     gws=(%zu,%zu) lws=(%zu,%zu)\n", cases[1].gws[0], cases[1].gws[1],
                cases[1].lws[0], cases[1].lws[1]);
    std::printf("stride_f4: gws=%zu lws=%zu nwg=%d (fixed, NOT n)\n\n", cases[2].gws[0],
                cases[2].lws[0], args.nwg);

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
    std::printf("  %-12s %10s %10s %10s %10s\n", "impl", "avg_ms", "best_ms", "vs_1d_n", "GB/s");
    for (int i = 0; i < 3; ++i) {
        const double gbs = gb_rw / (avgs[i] / 1e3);
        std::printf("  %-12s %10.3f %10.3f %9.2fx %10.1f\n", cases[i].name, avgs[i], bests[i],
                    avgs[0] / std::max(avgs[i], 1e-9), gbs);
    }

    static const int kNwgs[] = {32, 64, 128, 256, 512, 1024};
    std::printf("\n--- nwg sweep on stride_f4 (lws=%zu) ---\n", cases[2].lws[0]);
    std::printf("  %6s %10s %10s\n", "nwg", "avg_ms", "GB/s");
    double sweep_best = 1e300;
    int sweep_nwg = args.nwg;
    for (int nwg : kNwgs) {
        cases[2].gws[0] = cases[2].lws[0] * static_cast<size_t>(nwg);
        const auto [avg, best] = bench(cases[2]);
        (void)best;
        const double gbs = gb_rw / (avg / 1e3);
        std::printf("  %6d %10.3f %10.1f\n", nwg, avg, gbs);
        if (avg < sweep_best) {
            sweep_best = avg;
            sweep_nwg = nwg;
        }
    }
    std::printf("\nstride_f4 best: nwg=%d  avg=%.3f ms  (%.1f GB/s)\n", sweep_nwg, sweep_best,
                gb_rw / (sweep_best / 1e3));

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_dst);
    for (Case& c : cases) {
        clReleaseKernel(c.kn);
    }
    clReleaseProgram(p1d);
    clReleaseProgram(p2d);
    clReleaseProgram(p_stride);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
