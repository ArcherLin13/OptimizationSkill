// HarmonyOS OpenCL: findMaxValue — orig(2D 1px) vs baseline(1D 1px) vs opt(OCR-style).
// Default image 5760x4320. Kernel time only.
//
// Usage:
//   ./ocl_test_findmax [--width 5760] [--height 4320] [--runs 30]
//                       [--lwsx 16] [--lwsy 16] [--lws1d 256]
//                       [--lws-opt 256] [--nwg 256]

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
    std::string orig_path = "findmax_orig_2d.cl";
    std::string mine_path = "findmax_baseline.cl";
    std::string opt_path = "findmax_opt.cl";
    int width = 5760;
    int height = 4320;
    int runs = 30;
    int warmup = 5;
    int lwsx = 16;
    int lwsy = 16;
    int lws1d = 256;
    int lws_opt = 256;
    int nwg = 256;  // workgroups for opt (gws = lws_opt * nwg)
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--orig") == 0 && i + 1 < argc) {
            a.orig_path = argv[++i];
        } else if (std::strcmp(argv[i], "--mine") == 0 && i + 1 < argc) {
            a.mine_path = argv[++i];
        } else if (std::strcmp(argv[i], "--opt") == 0 && i + 1 < argc) {
            a.opt_path = argv[++i];
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
        } else if (std::strcmp(argv[i], "--lws1d") == 0 && i + 1 < argc) {
            a.lws1d = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lws-opt") == 0 && i + 1 < argc) {
            a.lws_opt = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nwg") == 0 && i + 1 < argc) {
            a.nwg = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: %s [--orig|--mine|--opt PATH] [--width W] [--height H]\n"
                "          [--lwsx 16] [--lwsy 16] [--lws1d 256]\n"
                "          [--lws-opt 256] [--nwg 256] [--runs N]\n",
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
    // Peak at last pixel so tiny images (1x1) still have a clear unique max.
    const size_t peak_i = n > 0 ? n - 1 : 0;
    for (size_t i = 0; i < n; ++i) {
        const int x = static_cast<int>(i % static_cast<size_t>(w));
        const int y = static_cast<int>(i / static_cast<size_t>(w));
        float v = static_cast<float>(((x * 13 + y * 7) % 1000)) / 1000.f;
        if (i == peak_i) {
            v = 12.5f;
        }
        img[i] = floatToHalf(v);
        ref_max = std::max(ref_max, v);
    }
}

struct KernelCase {
    const char* name;
    cl_kernel kn = nullptr;
    int dim = 1;
    size_t gws[2] = {0, 0};
    size_t lws[2] = {0, 0};
};

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

void setCommonArgs(cl_kernel kn, cl_mem buf_src, unsigned int w, unsigned int h, cl_mem buf_max) {
    OCL_CHECK(clSetKernelArg(kn, 0, sizeof(cl_mem), &buf_src), "arg0");
    OCL_CHECK(clSetKernelArg(kn, 1, sizeof(unsigned int), &w), "arg1");
    OCL_CHECK(clSetKernelArg(kn, 2, sizeof(unsigned int), &h), "arg2");
    OCL_CHECK(clSetKernelArg(kn, 3, sizeof(cl_mem), &buf_max), "arg3");
}

void enqueue(cl_command_queue queue, const KernelCase& c, cl_event* ev) {
    OCL_CHECK(clEnqueueNDRangeKernel(queue, c.kn, c.dim, nullptr, c.gws, c.lws, 0, nullptr, ev),
              c.name);
}

struct BenchResult {
    double avg_ms = 0.0;
    double best_ms = 0.0;
};

BenchResult runCase(cl_command_queue queue, cl_mem buf_max, KernelCase& c, float ref_max, int warmup,
                    int runs) {
    auto resetMax = [&] {
        uint32_t init = floatToHalf(-65504.0f);
        OCL_CHECK(clEnqueueWriteBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &init, 0,
                                       nullptr, nullptr),
                  "reset max");
    };

    resetMax();
    {
        cl_event ev = nullptr;
        enqueue(queue, c, &ev);
        OCL_CHECK(clWaitForEvents(1, &ev), "correctness wait");
        clReleaseEvent(ev);
    }
    uint32_t got_bits = 0;
    OCL_CHECK(clEnqueueReadBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &got_bits, 0, nullptr,
                                  nullptr),
              "read max");
    const float got = halfToFloat(static_cast<uint16_t>(got_bits & 0xffffu));
    const float abs_err = std::fabs(got - ref_max);
    std::printf("[%s] ref=%.6f gpu=%.6f err=%.3e %s\n", c.name, ref_max, got, abs_err,
                abs_err < 1e-2f ? "OK" : "FAIL");
    if (abs_err >= 1e-2f) {
        std::exit(1);
    }

    for (int i = 0; i < warmup; ++i) {
        resetMax();
        cl_event ev = nullptr;
        enqueue(queue, c, &ev);
        OCL_CHECK(clWaitForEvents(1, &ev), "warmup");
        clReleaseEvent(ev);
    }

    double sum = 0.0, best = 1e300;
    for (int i = 0; i < runs; ++i) {
        resetMax();
        OCL_CHECK(clFinish(queue), "finish");
        cl_event ev = nullptr;
        enqueue(queue, c, &ev);
        const double ms = profileMs(ev);
        clReleaseEvent(ev);
        sum += ms;
        best = std::min(best, ms);
    }
    return BenchResult{sum / runs, best};
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    const Args args = parseArgs(argc, argv);
    if (args.lwsx * args.lwsy > 256) {
        std::fprintf(stderr, "orig requires lwsx*lwsy <= 256\n");
        return 1;
    }
    if (args.lws_opt <= 0 || (args.lws_opt & (args.lws_opt - 1)) != 0) {
        std::fprintf(stderr, "--lws-opt must be power-of-two\n");
        return 1;
    }
    if (args.nwg <= 0) {
        std::fprintf(stderr, "--nwg must be > 0\n");
        return 1;
    }

#ifdef OCR_OPENCL_DLOPEN
    if (!opencl_load()) {
        return 1;
    }
#endif

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

    char opts_1d[128], opts_opt[128];
    std::snprintf(opts_1d, sizeof(opts_1d), "-cl-std=CL1.2 -DWG_SIZE=%d", args.lws1d);
    std::snprintf(opts_opt, sizeof(opts_opt), "-cl-std=CL1.2 -DLSIZE=%d", args.lws_opt);

    cl_program prog_orig = buildProgram(ctx, device, args.orig_path, "-cl-std=CL1.2");
    cl_program prog_mine = buildProgram(ctx, device, args.mine_path, opts_1d);
    cl_program prog_opt = buildProgram(ctx, device, args.opt_path, opts_opt);

    cl_kernel kn_opt = clCreateKernel(prog_opt, "findMaxValue", &err);
    OCL_CHECK(err, "opt kn");
    cl_mem buf_max = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    OCL_CHECK(err, "max");

    const size_t lws_opt = static_cast<size_t>(args.lws_opt);
    const size_t gws_opt = lws_opt * static_cast<size_t>(args.nwg);

    std::printf("=== findMaxValue ===\n");
    std::printf("device: %s\n", deviceName(device).c_str());
    std::printf("\n############################################################\n");
    std::printf("# SIZE SWEEP: fixed gws=%zu (lws %zu * nwg %d), NOT WxH\n", gws_opt, lws_opt,
                args.nwg);
    std::printf("# grid-stride: for (i=gid; i<n; i+=gws)  -- any image size OK\n");
    std::printf("############################################################\n\n");

    const struct {
        int w, h;
        const char* note;
    } kSizes[] = {
        {1, 1, "tiny"},
        {3, 5, "odd, n not multiple of 4"},
        {7, 9, "odd small"},
        {63, 63, "n not multiple of gws"},
        {64, 64, "small square"},
        {100, 100, "n < gws"},
        {640, 480, "VGA"},
        {1920, 1080, "FHD"},
        {1920, 1081, "odd height"},
        {4096, 1, "wide 1-row"},
        {1, 4096, "tall 1-col"},
        {5760, 4320, "default"},
        {args.width, args.height, "cli"},
    };

    std::printf("--- opt size sweep (same gws=%zu for all sizes) ---\n", gws_opt);
    int fail = 0;
    for (const auto& s : kSizes) {
        if (s.w <= 0 || s.h <= 0) {
            continue;
        }
        std::vector<uint16_t> img;
        float ref_max = 0.f;
        fillHalfImage(img, s.w, s.h, ref_max);
        const size_t nbytes = img.size() * sizeof(uint16_t);
        cl_mem buf_src =
            clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbytes, img.data(), &err);
        if (err != CL_SUCCESS) {
            std::printf("  %4dx%-4d  CREATE fail %d [%s]\n", s.w, s.h, err, s.note);
            ++fail;
            continue;
        }
        const unsigned wu = static_cast<unsigned>(s.w);
        const unsigned hu = static_cast<unsigned>(s.h);
        setCommonArgs(kn_opt, buf_src, wu, hu, buf_max);
        uint32_t init = floatToHalf(-65504.0f);
        OCL_CHECK(clEnqueueWriteBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &init, 0,
                                       nullptr, nullptr),
                  "reset");
        cl_event ev = nullptr;
        err = clEnqueueNDRangeKernel(queue, kn_opt, 1, nullptr, &gws_opt, &lws_opt, 0, nullptr, &ev);
        if (err != CL_SUCCESS) {
            std::printf("  %4dx%-4d  ENQUEUE fail %d n=%zu [%s]\n", s.w, s.h, err, img.size(),
                        s.note);
            ++fail;
            clReleaseMemObject(buf_src);
            continue;
        }
        err = clWaitForEvents(1, &ev);
        clReleaseEvent(ev);
        if (err != CL_SUCCESS) {
            std::printf("  %4dx%-4d  WAIT fail %d [%s]\n", s.w, s.h, err, s.note);
            ++fail;
            clReleaseMemObject(buf_src);
            continue;
        }
        uint32_t got_bits = 0;
        OCL_CHECK(clEnqueueReadBuffer(queue, buf_max, CL_TRUE, 0, sizeof(uint32_t), &got_bits, 0,
                                      nullptr, nullptr),
                  "read");
        const float got = halfToFloat(static_cast<uint16_t>(got_bits & 0xffffu));
        const float abs_err = std::fabs(got - ref_max);
        const bool ok = abs_err < 1e-2f;
        std::printf("  %4dx%-4d n=%8zu ref=%.4f gpu=%.4f err=%.2e %s [%s]\n", s.w, s.h, img.size(),
                    ref_max, got, abs_err, ok ? "OK" : "FAIL", s.note);
        if (!ok) {
            ++fail;
        }
        clReleaseMemObject(buf_src);
    }
    if (fail) {
        std::fprintf(stderr, "size sweep FAIL=%d\n", fail);
        return 1;
    }
    std::printf("size sweep: all OK\n\n");

    const int W = args.width;
    const int H = args.height;
    const size_t nPix = static_cast<size_t>(W) * static_cast<size_t>(H);
    const size_t bytes = nPix * sizeof(uint16_t);

    std::vector<uint16_t> host;
    float ref_max = 0.f;
    fillHalfImage(host, W, H, ref_max);

    cl_mem buf_src =
        clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host.data(), &err);
    OCL_CHECK(err, "src");

    const unsigned int wu = static_cast<unsigned int>(W);
    const unsigned int hu = static_cast<unsigned int>(H);

    KernelCase orig;
    orig.name = "orig_2d_1px";
    orig.kn = clCreateKernel(prog_orig, "findMaxValue", &err);
    OCL_CHECK(err, "orig kn");
    orig.dim = 2;
    orig.lws[0] = static_cast<size_t>(args.lwsx);
    orig.lws[1] = static_cast<size_t>(args.lwsy);
    orig.gws[0] = ((static_cast<size_t>(W) + orig.lws[0] - 1) / orig.lws[0]) * orig.lws[0];
    orig.gws[1] = ((static_cast<size_t>(H) + orig.lws[1] - 1) / orig.lws[1]) * orig.lws[1];
    setCommonArgs(orig.kn, buf_src, wu, hu, buf_max);

    KernelCase mine;
    mine.name = "base_1d_1px";
    mine.kn = clCreateKernel(prog_mine, "findMaxValue", &err);
    OCL_CHECK(err, "mine kn");
    mine.dim = 1;
    mine.lws[0] = static_cast<size_t>(args.lws1d);
    mine.gws[0] = ((nPix + mine.lws[0] - 1) / mine.lws[0]) * mine.lws[0];
    setCommonArgs(mine.kn, buf_src, wu, hu, buf_max);

    KernelCase opt;
    opt.name = "opt_stride";
    opt.kn = kn_opt;
    opt.dim = 1;
    opt.lws[0] = lws_opt;
    opt.gws[0] = gws_opt;
    setCommonArgs(opt.kn, buf_src, wu, hu, buf_max);

    const double elems_per_wi = static_cast<double>(nPix) / static_cast<double>(opt.gws[0]);
    std::printf("--- timed bench %dx%d ---\n", W, H);
    std::printf("orig gws=(%zu,%zu)  base gws=%zu  opt gws=%zu (~%.1f px/WI)\n\n", orig.gws[0],
                orig.gws[1], mine.gws[0], opt.gws[0], elems_per_wi);

    const BenchResult r_orig = runCase(queue, buf_max, orig, ref_max, args.warmup, args.runs);
    const BenchResult r_mine = runCase(queue, buf_max, mine, ref_max, args.warmup, args.runs);
    const BenchResult r_opt = runCase(queue, buf_max, opt, ref_max, args.warmup, args.runs);

    const double gb = bytes / 1e9;
    std::printf("\n--- kernel time (avg %d) ---\n", args.runs);
    std::printf("  %-14s %10s %10s %10s\n", "impl", "avg_ms", "best_ms", "vs_orig");
    auto row = [&](const char* name, const BenchResult& r) {
        std::printf("  %-14s %10.3f %10.3f %9.2fx\n", name, r.avg_ms, r.best_ms,
                    r_orig.avg_ms / std::max(r.avg_ms, 1e-9));
        (void)gb;
    };
    row("orig_2d_1px", r_orig);
    row("base_1d_1px", r_mine);
    row("opt_stride", r_opt);

    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_max);
    clReleaseKernel(orig.kn);
    clReleaseKernel(mine.kn);
    clReleaseKernel(opt.kn);
    clReleaseProgram(prog_orig);
    clReleaseProgram(prog_mine);
    clReleaseProgram(prog_opt);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
