// Baseline: orig findMaxValue + enhanceBrightness (2 kernels)
// Opt:     findMaxAndEnhance fused (1 kernel, based on opt_stride)
// Default: 5760x4320 half image. Kernel time only. Correctness vs CPU.
//
// Usage:
//   ./ocl_test_enhance [--width 5760] [--height 4320] [--runs 30]
//                       [--lwsx 16] [--lwsy 16] [--lws-opt 256] [--nwg 256]

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
    std::string findmax_path = "findmax_orig_2d.cl";
    std::string enhance_path = "enhance_brightness.cl";
    std::string fused_path = "findmax_enhance_fused.cl";
    int width = 5760;
    int height = 4320;
    int runs = 30;
    int warmup = 5;
    int lwsx = 16;
    int lwsy = 16;
    int lws_opt = 256;
    int nwg = 256;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--findmax") == 0 && i + 1 < argc) {
            a.findmax_path = argv[++i];
        } else if (std::strcmp(argv[i], "--enhance") == 0 && i + 1 < argc) {
            a.enhance_path = argv[++i];
        } else if (std::strcmp(argv[i], "--fused") == 0 && i + 1 < argc) {
            a.fused_path = argv[++i];
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
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
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--width W] [--height H] [--lws-opt N] [--nwg N] [--runs N]\n",
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

void fillHalfImage(std::vector<uint16_t>& img, int w, int h, float& ref_max) {
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    img.resize(n);
    ref_max = -1e30f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = static_cast<float>(((x * 13 + y * 7) % 1000)) / 1000.f;
            if (x == w * 2 / 3 && y == h * 3 / 5) {
                v = 12.5f;
            }
            img[static_cast<size_t>(y) * w + x] = floatToHalf(v);
            ref_max = std::max(ref_max, v);
        }
    }
}

void cpuEnhance(std::vector<uint16_t>& img, float ref_max) {
    const float div = std::min(1.0f, ref_max);
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

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    if (args.lwsx * args.lwsy > 256) {
        std::fprintf(stderr, "orig findmax requires lwsx*lwsy <= 256\n");
        return 1;
    }
    if (args.lws_opt <= 0 || (args.lws_opt & (args.lws_opt - 1)) != 0 || args.nwg <= 0) {
        std::fprintf(stderr, "bad --lws-opt / --nwg\n");
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

    std::vector<uint16_t> host_src;
    float ref_max = 0.f;
    fillHalfImage(host_src, W, H, ref_max);
    std::vector<uint16_t> host_ref = host_src;
    cpuEnhance(host_ref, ref_max);

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

    char opts_fused[128];
    std::snprintf(opts_fused, sizeof(opts_fused), "-cl-std=CL1.2 -DLSIZE=%d", args.lws_opt);

    cl_program prog_fm = buildProgram(ctx, device, args.findmax_path, "-cl-std=CL1.2");
    cl_program prog_en = buildProgram(ctx, device, args.enhance_path, "-cl-std=CL1.2");
    cl_program prog_fu = buildProgram(ctx, device, args.fused_path, opts_fused);

    cl_kernel kn_fm = clCreateKernel(prog_fm, "findMaxValue", &err);
    OCL_CHECK(err, "findMaxValue");
    cl_kernel kn_en = clCreateKernel(prog_en, "enhanceBrightness", &err);
    OCL_CHECK(err, "enhanceBrightness");
    cl_kernel kn_fu = clCreateKernel(prog_fu, "findMaxAndEnhance", &err);
    OCL_CHECK(err, "findMaxAndEnhance");

    cl_mem buf_src = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    OCL_CHECK(err, "buf_src");
    cl_mem buf_max = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    OCL_CHECK(err, "buf_max");
    cl_mem buf_sync = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_int) * 2, nullptr, &err);
    OCL_CHECK(err, "buf_sync");

    const unsigned int wu = static_cast<unsigned int>(W);
    const unsigned int hu = static_cast<unsigned int>(H);

    OCL_CHECK(clSetKernelArg(kn_fm, 0, sizeof(cl_mem), &buf_src), "fm0");
    OCL_CHECK(clSetKernelArg(kn_fm, 1, sizeof(unsigned int), &wu), "fm1");
    OCL_CHECK(clSetKernelArg(kn_fm, 2, sizeof(unsigned int), &hu), "fm2");
    OCL_CHECK(clSetKernelArg(kn_fm, 3, sizeof(cl_mem), &buf_max), "fm3");

    OCL_CHECK(clSetKernelArg(kn_en, 0, sizeof(cl_mem), &buf_src), "en0");
    OCL_CHECK(clSetKernelArg(kn_en, 1, sizeof(unsigned int), &wu), "en1");
    OCL_CHECK(clSetKernelArg(kn_en, 2, sizeof(unsigned int), &hu), "en2");
    OCL_CHECK(clSetKernelArg(kn_en, 3, sizeof(cl_mem), &buf_max), "en3");

    OCL_CHECK(clSetKernelArg(kn_fu, 0, sizeof(cl_mem), &buf_src), "fu0");
    OCL_CHECK(clSetKernelArg(kn_fu, 1, sizeof(unsigned int), &wu), "fu1");
    OCL_CHECK(clSetKernelArg(kn_fu, 2, sizeof(unsigned int), &hu), "fu2");
    OCL_CHECK(clSetKernelArg(kn_fu, 3, sizeof(cl_mem), &buf_max), "fu3");
    OCL_CHECK(clSetKernelArg(kn_fu, 4, sizeof(cl_mem), &buf_sync), "fu4");

    size_t gws2[2] = {((size_t)W + args.lwsx - 1) / args.lwsx * args.lwsx,
                      ((size_t)H + args.lwsy - 1) / args.lwsy * args.lwsy};
    size_t lws2[2] = {(size_t)args.lwsx, (size_t)args.lwsy};
    size_t gws1 = (size_t)args.lws_opt * (size_t)args.nwg;
    size_t lws1 = (size_t)args.lws_opt;

    auto uploadSrc = [&] {
        OCL_CHECK(clEnqueueWriteBuffer(queue, buf_src, CL_TRUE, 0, bytes, host_src.data(), 0, nullptr,
                                       nullptr),
                  "upload src");
    };
    auto resetMax = [&] {
        uint32_t init = floatToHalf(-65504.0f);
        OCL_CHECK(clEnqueueWriteBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &init, 0,
                                       nullptr, nullptr),
                  "reset max");
    };
    auto resetSync = [&] {
        int z[2] = {0, 0};
        OCL_CHECK(clEnqueueWriteBuffer(queue, buf_sync, CL_TRUE, 0, sizeof(z), z, 0, nullptr, nullptr),
                  "reset sync");
    };

    std::vector<uint16_t> out(nPix);

    auto checkOut = [&](const char* tag) {
        OCL_CHECK(clEnqueueReadBuffer(queue, buf_src, CL_TRUE, 0, bytes, out.data(), 0, nullptr,
                                      nullptr),
                  "read out");
        const float diff = maxAbsDiffHalf(out, host_ref);
        std::printf("[%s] vs CPU enhance: max_abs_diff=%.3e %s\n", tag, diff,
                    diff < 2e-2f ? "OK" : "FAIL");
        if (diff >= 2e-2f) {
            std::exit(1);
        }
    };

    std::printf("##############################################################\n");
    std::printf("# PIPELINE: findMaxValue + enhanceBrightness (NOT findmax-only)\n");
    std::printf("#   enhance: divisor=fmin(1,max);  src[i] *= 1/divisor\n");
    std::printf("#   A) baseline = 2 kernels (orig findmax THEN enhance)\n");
    std::printf("#   B) fused    = 1 kernel  (findMaxAndEnhance)\n");
    std::printf("##############################################################\n");
    std::printf("device: %s\n", deviceName(device).c_str());
    std::printf("size:   %d x %d  ref_max=%.4f  divisor=fmin(1,max)=%.4f\n", W, H, ref_max,
                std::min(1.0f, ref_max));
    std::printf("\n[A] baseline 2-kernel path:\n");
    std::printf("    1) findMaxValue     <- %s\n", args.findmax_path.c_str());
    std::printf("    2) enhanceBrightness<- %s\n", args.enhance_path.c_str());
    std::printf("    launch 2D gws=(%zu,%zu) lws=(%zu,%zu)\n", gws2[0], gws2[1], lws2[0], lws2[1]);
    std::printf("\n[B] fused 1-kernel path:\n");
    std::printf("    1) findMaxAndEnhance <- %s\n", args.fused_path.c_str());
    std::printf("    launch 1D gws=%zu lws=%zu nwg=%d\n\n", gws1, lws1, args.nwg);

    // Correctness
    std::printf("--- correctness (full image after enhance vs CPU) ---\n");
    uploadSrc();
    resetMax();
    {
        cl_event e0 = nullptr, e1 = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fm, 2, nullptr, gws2, lws2, 0, nullptr, &e0),
                  "fm corr");
        OCL_CHECK(clWaitForEvents(1, &e0), "fm wait");
        clReleaseEvent(e0);
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_en, 2, nullptr, gws2, lws2, 0, nullptr, &e1),
                  "en corr");
        OCL_CHECK(clWaitForEvents(1, &e1), "en wait");
        clReleaseEvent(e1);
    }
    checkOut("A_baseline findmax+enhance");

    uploadSrc();
    resetMax();
    resetSync();
    {
        cl_event e = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fu, 1, nullptr, &gws1, &lws1, 0, nullptr, &e),
                  "fu corr");
        OCL_CHECK(clWaitForEvents(1, &e), "fu wait");
        clReleaseEvent(e);
    }
    checkOut("B_fused    findMaxAndEnhance");
    std::printf("\n");

    auto benchBaseline = [&](int warmup, int runs) {
        for (int i = 0; i < warmup; ++i) {
            uploadSrc();
            resetMax();
            cl_event e0 = nullptr, e1 = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fm, 2, nullptr, gws2, lws2, 0, nullptr, &e0),
                      "fm w");
            OCL_CHECK(clWaitForEvents(1, &e0), "w0");
            clReleaseEvent(e0);
            OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_en, 2, nullptr, gws2, lws2, 0, nullptr, &e1),
                      "en w");
            OCL_CHECK(clWaitForEvents(1, &e1), "w1");
            clReleaseEvent(e1);
        }
        double sum_fm = 0, sum_en = 0, best = 1e300;
        for (int i = 0; i < runs; ++i) {
            uploadSrc();
            resetMax();
            OCL_CHECK(clFinish(queue), "fin");
            cl_event e0 = nullptr, e1 = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fm, 2, nullptr, gws2, lws2, 0, nullptr, &e0),
                      "fm");
            const double ms0 = profileMs(e0);
            clReleaseEvent(e0);
            OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_en, 2, nullptr, gws2, lws2, 0, nullptr, &e1),
                      "en");
            const double ms1 = profileMs(e1);
            clReleaseEvent(e1);
            sum_fm += ms0;
            sum_en += ms1;
            best = std::min(best, ms0 + ms1);
        }
        return std::tuple<double, double, double>{sum_fm / runs, sum_en / runs, best};
    };

    auto benchFused = [&](int warmup, int runs) {
        for (int i = 0; i < warmup; ++i) {
            uploadSrc();
            resetMax();
            resetSync();
            cl_event e = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fu, 1, nullptr, &gws1, &lws1, 0, nullptr, &e),
                      "fu w");
            OCL_CHECK(clWaitForEvents(1, &e), "fu ww");
            clReleaseEvent(e);
        }
        double sum = 0, best = 1e300;
        for (int i = 0; i < runs; ++i) {
            uploadSrc();
            resetMax();
            resetSync();
            OCL_CHECK(clFinish(queue), "fin");
            cl_event e = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fu, 1, nullptr, &gws1, &lws1, 0, nullptr, &e),
                      "fu");
            const double ms = profileMs(e);
            clReleaseEvent(e);
            sum += ms;
            best = std::min(best, ms);
        }
        return std::pair<double, double>{sum / runs, best};
    };

    const auto [fm_ms, en_ms, base_best] = benchBaseline(args.warmup, args.runs);
    const auto [fu_ms, fu_best] = benchFused(args.warmup, args.runs);
    const double base_avg = fm_ms + en_ms;

    std::printf("\n--- PIPELINE kernel time (avg %d runs) ---\n", args.runs);
    std::printf("  [A] findMaxValue (orig)     %.3f ms\n", fm_ms);
    std::printf("  [A] enhanceBrightness       %.3f ms\n", en_ms);
    std::printf("  [A] TOTAL 2-kernel           %.3f ms  (best %.3f)\n", base_avg, base_best);
    std::printf("  [B] findMaxAndEnhance fused %.3f ms  (best %.3f)\n", fu_ms, fu_best);
    std::printf("  speedup [A]/[B]:            %.2fx\n", base_avg / std::max(fu_ms, 1e-9));
    std::printf("##############################################################\n");
    std::printf("# end PIPELINE bench (findmax + enhance)\n");
    std::printf("##############################################################\n");

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_max);
    clReleaseMemObject(buf_sync);
    clReleaseKernel(kn_fm);
    clReleaseKernel(kn_en);
    clReleaseKernel(kn_fu);
    clReleaseProgram(prog_fm);
    clReleaseProgram(prog_en);
    clReleaseProgram(prog_fu);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
