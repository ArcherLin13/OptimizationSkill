// Baseline: orig findMax + enhanceBrightness (2 kernels, 2D 1px)
// Fused:   findMaxAndEnhance 1-kernel, SINGLE workgroup (no cross-WG spin)
// Opt:     findmax_opt + enhance_opt (2 kernels, grid-stride/half4) — fast path
//
// Note: multi-WG grid-sync fused caused CL_-14 on device; removed.

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
    std::string findmax_orig = "findmax_orig_2d.cl";
    std::string enhance_base = "enhance_brightness.cl";
    std::string fused_path = "findmax_enhance_fused.cl";
    std::string findmax_opt = "findmax_opt.cl";
    std::string enhance_opt = "enhance_brightness_opt.cl";
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
        auto take = [&](const char* key, std::string& dst) {
            if (std::strcmp(argv[i], key) == 0 && i + 1 < argc) {
                dst = argv[++i];
                return true;
            }
            return false;
        };
        if (take("--findmax", a.findmax_orig) || take("--enhance", a.enhance_base) ||
            take("--fused", a.fused_path) || take("--findmax-opt", a.findmax_opt) ||
            take("--enhance-opt", a.enhance_opt)) {
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
    cl_int st = CL_SUCCESS;
    OCL_CHECK(clWaitForEvents(1, &ev), "clWaitForEvents");
    clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, nullptr);
    if (st < 0) {
        std::fprintf(stderr, "event exec status %d (negative = error)\n", st);
        std::exit(1);
    }
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
    // Peak MUST be < 1 so enhance actually scales (divisor=max, inv=1/max).
    // Old peak 12.5 made divisor=1 → enhance no-op → false OK even if enhance never ran.
    const float kPeak = 0.25f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = static_cast<float>(((x * 13 + y * 7) % 1000)) / 1000.f;  // [0,1)
            if (v >= kPeak) {
                v = kPeak * 0.5f;  // keep unique global max at injected pixel
            }
            if (x == w * 2 / 3 && y == h * 3 / 5) {
                v = kPeak;
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

float meanAbsDiffHalf(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += std::fabs(halfToFloat(a[i]) - halfToFloat(b[i]));
    }
    return static_cast<float>(s / static_cast<double>(a.size()));
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

void set4(cl_kernel kn, cl_mem src, unsigned w, unsigned h, cl_mem mx) {
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &src), "a0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(unsigned), &w), "a1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(unsigned), &h), "a2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &mx), "a3");
}

void set5(cl_kernel kn, cl_mem src, unsigned w, unsigned h, cl_mem mx, cl_mem sync) {
    set4(kn, src, w, h, mx);
    OCL_CHECK(clSetKernelArg(kn, 4, sizeof(cl_mem), &sync), "a4");
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

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

    char opts_lsize[128];
    std::snprintf(opts_lsize, sizeof(opts_lsize), "-cl-std=CL1.2 -DLSIZE=%d", args.lws_opt);

    cl_program p_fm = buildProgram(ctx, device, args.findmax_orig, "-cl-std=CL1.2");
    cl_program p_en = buildProgram(ctx, device, args.enhance_base, "-cl-std=CL1.2");
    cl_program p_fu = buildProgram(ctx, device, args.fused_path, opts_lsize);
    cl_program p_fmo = buildProgram(ctx, device, args.findmax_opt, opts_lsize);
    cl_program p_eno = buildProgram(ctx, device, args.enhance_opt, opts_lsize);

    cl_kernel kn_fm = clCreateKernel(p_fm, "findMaxValue", &err);
    OCL_CHECK(err, "findMaxValue orig");
    cl_kernel kn_en = clCreateKernel(p_en, "enhanceBrightness", &err);
    OCL_CHECK(err, "enhance base");
    cl_kernel kn_fu = clCreateKernel(p_fu, "findMaxAndEnhance", &err);
    OCL_CHECK(err, "fused");
    cl_kernel kn_fmo = clCreateKernel(p_fmo, "findMaxValue", &err);
    OCL_CHECK(err, "findMax opt");
    cl_kernel kn_eno = clCreateKernel(p_eno, "enhanceBrightness", &err);
    OCL_CHECK(err, "enhance opt");

    cl_mem buf_src = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    OCL_CHECK(err, "buf_src");
    cl_mem buf_gold =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_src.data(), &err);
    OCL_CHECK(err, "buf_gold");
    cl_mem buf_max = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    OCL_CHECK(err, "buf_max");
    cl_mem buf_sync = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_int) * 2, nullptr, &err);
    OCL_CHECK(err, "buf_sync");

    const unsigned wu = (unsigned)W, hu = (unsigned)H;
    set4(kn_fm, buf_src, wu, hu, buf_max);
    set4(kn_en, buf_src, wu, hu, buf_max);
    set5(kn_fu, buf_src, wu, hu, buf_max, buf_sync);
    set4(kn_fmo, buf_src, wu, hu, buf_max);
    set4(kn_eno, buf_src, wu, hu, buf_max);

    size_t gws2[2] = {((size_t)W + args.lwsx - 1) / args.lwsx * args.lwsx,
                      ((size_t)H + args.lwsy - 1) / args.lwsy * args.lwsy};
    size_t lws2[2] = {(size_t)args.lwsx, (size_t)args.lwsy};
    // Fused multi-WG: lid0-only grid sync (all-lane spin caused CL_-14).
    size_t lws_fu = (size_t)args.lws_opt;
    size_t gws_fu = lws_fu * (size_t)args.nwg;
    // Opt 2-kernel: multi-WG stride
    size_t lws_o = (size_t)args.lws_opt;
    size_t gws_o = lws_o * (size_t)args.nwg;

    auto restoreSrc = [&] {
        OCL_CHECK(clEnqueueCopyBuffer(queue, buf_gold, buf_src, 0, 0, bytes, 0, nullptr, nullptr),
                  "restore");
        OCL_CHECK(clFinish(queue), "restore fin");
    };
    auto resetMax = [&] {
        uint32_t init = floatToHalf(-65504.0f);
        OCL_CHECK(clEnqueueWriteBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &init, 0,
                                       nullptr, nullptr),
                  "reset max");
    };
    auto resetSync = [&] {
        int z[2] = {0, 0};
        OCL_CHECK(
            clEnqueueWriteBuffer(queue, buf_sync, CL_TRUE, 0, sizeof(z), z, 0, nullptr, nullptr),
            "reset sync");
    };

    std::vector<uint16_t> out(nPix);
    auto checkOut = [&](const char* tag) {
        // 1) max_value must match ref_max
        uint32_t max_bits = 0;
        OCL_CHECK(clEnqueueReadBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &max_bits, 0,
                                      nullptr, nullptr),
                  "read max");
        const float gpu_max = halfToFloat(static_cast<uint16_t>(max_bits & 0xffffu));
        const float max_err = std::fabs(gpu_max - ref_max);

        // 2) image vs CPU enhanced ref
        OCL_CHECK(clEnqueueReadBuffer(queue, buf_src, CL_TRUE, 0, bytes, out.data(), 0, nullptr,
                                      nullptr),
                  "read src");
        const float vs_ref = maxAbsDiffHalf(out, host_ref);
        const float vs_in = meanAbsDiffHalf(out, host_src);
        const float div = std::min(1.0f, ref_max);
        const bool must_scale = div < 1.0f - 1e-6f;

        const bool ok_max = max_err < 1e-2f;
        const bool ok_img = vs_ref < 2e-2f;
        // If max<1, enhance must change the image; otherwise the check is vacuous.
        const bool ok_changed = !must_scale || (vs_in > 1e-3f);

        std::printf("[%s]\n", tag);
        std::printf("    gpu_max=%.6f ref_max=%.6f max_err=%.3e %s\n", gpu_max, ref_max, max_err,
                    ok_max ? "OK" : "FAIL");
        std::printf("    vs CPU enhance: max_abs_diff=%.3e %s\n", vs_ref, ok_img ? "OK" : "FAIL");
        std::printf("    vs input (mean abs): %.3e  (divisor=%.4f, must_scale=%s) %s\n", vs_in, div,
                    must_scale ? "yes" : "no", ok_changed ? "OK" : "FAIL(no-op?)");
        if (!ok_max || !ok_img || !ok_changed) {
            std::fprintf(stderr, "CORRECTNESS FAIL on %s\n", tag);
            std::exit(1);
        }
    };

    std::printf("##############################################################\n");
    std::printf("# PIPELINE: findMax + enhanceBrightness\n");
    std::printf("#   A) baseline 2k: orig findmax + enhance (2D 1px)\n");
    std::printf("#   B) fused 1k:    multi-WG + lid0-only grid sync + enhance\n");
    std::printf("#   C) opt 2k:      findmax_opt + enhance_opt\n");
    std::printf("##############################################################\n");
    std::printf("device: %s\n", deviceName(device).c_str());
    std::printf("size: %d x %d  ref_max=%.4f divisor=%.4f inv=%.4f  (max<1 so enhance is NOT no-op)\n",
                W, H, ref_max, std::min(1.f, ref_max), 1.f / std::min(1.f, ref_max));
    std::printf("[A] gws=(%zu,%zu) lws=(%zu,%zu)\n", gws2[0], gws2[1], lws2[0], lws2[1]);
    std::printf("[B] gws=%zu lws=%zu nwg=%d (fused multi-WG)\n", gws_fu, lws_fu, args.nwg);
    std::printf("[C] gws=%zu lws=%zu nwg=%d\n\n", gws_o, lws_o, args.nwg);

    std::printf("--- correctness ---\n");
    restoreSrc();
    resetMax();
    {
        cl_event e0 = nullptr, e1 = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fm, 2, nullptr, gws2, lws2, 0, nullptr, &e0), "A0");
        OCL_CHECK(clWaitForEvents(1, &e0), "A0w");
        clReleaseEvent(e0);
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_en, 2, nullptr, gws2, lws2, 0, nullptr, &e1), "A1");
        OCL_CHECK(clWaitForEvents(1, &e1), "A1w");
        clReleaseEvent(e1);
    }
    checkOut("A baseline");

    restoreSrc();
    resetMax();
    resetSync();
    {
        cl_event e = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fu, 1, nullptr, &gws_fu, &lws_fu, 0, nullptr, &e),
                  "B");
        OCL_CHECK(clWaitForEvents(1, &e), "Bw");
        clReleaseEvent(e);
    }
    checkOut("B fused multi-WG");

    restoreSrc();
    resetMax();
    {
        cl_event e0 = nullptr, e1 = nullptr;
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_fmo, 1, nullptr, &gws_o, &lws_o, 0, nullptr, &e0),
                  "C0");
        OCL_CHECK(clWaitForEvents(1, &e0), "C0w");
        clReleaseEvent(e0);
        OCL_CHECK(clEnqueueNDRangeKernel(queue, kn_eno, 1, nullptr, &gws_o, &lws_o, 0, nullptr, &e1),
                  "C1");
        OCL_CHECK(clWaitForEvents(1, &e1), "C1w");
        clReleaseEvent(e1);
    }
    checkOut("C opt 2k");
    std::printf("correctness done. starting TIMING...\n\n");

    auto time2 = [&](const char* tag, cl_kernel k0, int dim0, const size_t* g0, const size_t* l0,
                     cl_kernel k1, int dim1, const size_t* g1, const size_t* l1) {
        std::printf(">>> TIMING %s (%d+%d)...\n", tag, args.warmup, args.runs);
        for (int i = 0; i < args.warmup; ++i) {
            restoreSrc();
            resetMax();
            cl_event e0 = nullptr, e1 = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, k0, dim0, nullptr, g0, l0, 0, nullptr, &e0), "w0");
            OCL_CHECK(clWaitForEvents(1, &e0), "ww0");
            clReleaseEvent(e0);
            OCL_CHECK(clEnqueueNDRangeKernel(queue, k1, dim1, nullptr, g1, l1, 0, nullptr, &e1), "w1");
            OCL_CHECK(clWaitForEvents(1, &e1), "ww1");
            clReleaseEvent(e1);
        }
        double s0 = 0, s1 = 0, best = 1e300;
        for (int i = 0; i < args.runs; ++i) {
            restoreSrc();
            resetMax();
            OCL_CHECK(clFinish(queue), "fin");
            cl_event e0 = nullptr, e1 = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, k0, dim0, nullptr, g0, l0, 0, nullptr, &e0), "t0");
            const double m0 = profileMs(e0);
            clReleaseEvent(e0);
            OCL_CHECK(clEnqueueNDRangeKernel(queue, k1, dim1, nullptr, g1, l1, 0, nullptr, &e1), "t1");
            const double m1 = profileMs(e1);
            clReleaseEvent(e1);
            s0 += m0;
            s1 += m1;
            best = std::min(best, m0 + m1);
            if ((i + 1) == 1 || (i + 1) == args.runs || ((i + 1) % 10) == 0) {
                std::printf("    %s run %d/%d  k0=%.3f k1=%.3f sum=%.3f\n", tag, i + 1, args.runs, m0,
                            m1, m0 + m1);
            }
        }
        const double a0 = s0 / args.runs, a1 = s1 / args.runs;
        std::printf(">>> %s DONE  k0=%.3f k1=%.3f TOTAL=%.3f (best %.3f)\n\n", tag, a0, a1, a0 + a1,
                    best);
        return std::tuple<double, double, double>{a0, a1, best};
    };

    auto time1 = [&](const char* tag, cl_kernel k, const size_t* g, const size_t* l) {
        std::printf(">>> TIMING %s (%d+%d)...\n", tag, args.warmup, args.runs);
        for (int i = 0; i < args.warmup; ++i) {
            restoreSrc();
            resetMax();
            resetSync();
            cl_event e = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, k, 1, nullptr, g, l, 0, nullptr, &e), "w");
            OCL_CHECK(clWaitForEvents(1, &e), "ww");
            clReleaseEvent(e);
        }
        double sum = 0, best = 1e300;
        for (int i = 0; i < args.runs; ++i) {
            restoreSrc();
            resetMax();
            resetSync();
            OCL_CHECK(clFinish(queue), "fin");
            cl_event e = nullptr;
            OCL_CHECK(clEnqueueNDRangeKernel(queue, k, 1, nullptr, g, l, 0, nullptr, &e), "t");
            const double ms = profileMs(e);
            clReleaseEvent(e);
            sum += ms;
            best = std::min(best, ms);
            if ((i + 1) == 1 || (i + 1) == args.runs || ((i + 1) % 10) == 0) {
                std::printf("    %s run %d/%d  fused=%.3f\n", tag, i + 1, args.runs, ms);
            }
        }
        const double avg = sum / args.runs;
        std::printf(">>> %s DONE  fused=%.3f (best %.3f)\n\n", tag, avg, best);
        return std::pair<double, double>{avg, best};
    };

    const auto [a0, a1, a_best] = time2("[A]", kn_fm, 2, gws2, lws2, kn_en, 2, gws2, lws2);
    const auto [b_avg, b_best] = time1("[B]", kn_fu, &gws_fu, &lws_fu);
    const auto [c0, c1, c_best] = time2("[C]", kn_fmo, 1, &gws_o, &lws_o, kn_eno, 1, &gws_o, &lws_o);
    const double a_tot = a0 + a1;
    const double c_tot = c0 + c1;

    std::printf("==============================================================\n");
    std::printf(" PIPELINE SUMMARY (ms)\n");
    std::printf("  [A] baseline findmax+enhance  %.3f + %.3f = %.3f\n", a0, a1, a_tot);
    std::printf("  [B] fused 1-kernel multi-WG   %.3f\n", b_avg);
    std::printf("  [C] opt findmax+enhance       %.3f + %.3f = %.3f\n", c0, c1, c_tot);
    std::printf("  A/B=%.2fx  A/C=%.2fx  ( >1 means faster than baseline )\n",
                a_tot / std::max(b_avg, 1e-9), a_tot / std::max(c_tot, 1e-9));
    std::printf("==============================================================\n");

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_gold);
    clReleaseMemObject(buf_max);
    clReleaseMemObject(buf_sync);
    clReleaseKernel(kn_fm);
    clReleaseKernel(kn_en);
    clReleaseKernel(kn_fu);
    clReleaseKernel(kn_fmo);
    clReleaseKernel(kn_eno);
    clReleaseProgram(p_fm);
    clReleaseProgram(p_en);
    clReleaseProgram(p_fu);
    clReleaseProgram(p_fmo);
    clReleaseProgram(p_eno);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
