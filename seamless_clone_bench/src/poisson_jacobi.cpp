#include "poisson_jacobi.h"

#include "seamless_roi.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__OHOS__) || defined(__ANDROID__) || defined(__linux__)
#define POISSON_JACOBI_DLOPEN 1
#include <dlfcn.h>
#endif

namespace {

void computeGradientX(const cv::Mat& img, cv::Mat& gx) {
    const cv::Mat kernel = (cv::Mat_<char>(1, 3) << 0, -1, 1);
    cv::filter2D(img, gx, CV_32F, kernel);
    if (img.channels() == 1) {
        cv::cvtColor(gx, gx, cv::COLOR_GRAY2BGR);
    }
}

void computeGradientY(const cv::Mat& img, cv::Mat& gy) {
    const cv::Mat kernel = (cv::Mat_<char>(3, 1) << 0, -1, 1);
    cv::filter2D(img, gy, CV_32F, kernel);
    if (img.channels() == 1) {
        cv::cvtColor(gy, gy, cv::COLOR_GRAY2BGR);
    }
}

void computeLaplacianX(const cv::Mat& img, cv::Mat& lx) {
    const cv::Mat kernel = (cv::Mat_<char>(1, 3) << -1, 1, 0);
    cv::filter2D(img, lx, CV_32F, kernel);
}

void computeLaplacianY(const cv::Mat& img, cv::Mat& ly) {
    const cv::Mat kernel = (cv::Mat_<char>(3, 1) << -1, 1, 0);
    cv::filter2D(img, ly, CV_32F, kernel);
}

void buildNormalCloneLaplacian(const cv::Mat& destination, const cv::Mat& patch, cv::Mat binaryMask,
                               std::vector<cv::Mat>& lapPerChannel) {
    cv::Mat destGx;
    cv::Mat destGy;
    cv::Mat patchGx;
    cv::Mat patchGy;
    computeGradientX(destination, destGx);
    computeGradientY(destination, destGy);
    computeGradientX(patch, patchGx);
    computeGradientY(patch, patchGy);

    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::erode(binaryMask, binaryMask, kernel, cv::Point(-1, -1), 3);

    cv::Mat maskFloat;
    binaryMask.convertTo(maskFloat, CV_32FC1, 1.0 / 255.0);

    cv::Mat invMask;
    cv::bitwise_not(binaryMask, invMask);
    cv::Mat invMaskFloat;
    invMask.convertTo(invMaskFloat, CV_32FC1, 1.0 / 255.0);

    std::vector<cv::Mat> patchGxCh;
    std::vector<cv::Mat> patchGyCh;
    std::vector<cv::Mat> destGxCh;
    std::vector<cv::Mat> destGyCh;
    cv::split(patchGx, patchGxCh);
    cv::split(patchGy, patchGyCh);
    cv::split(destGx, destGxCh);
    cv::split(destGy, destGyCh);

    lapPerChannel.resize(3);
    for (int c = 0; c < 3; ++c) {
        cv::multiply(patchGxCh[c], maskFloat, patchGxCh[c]);
        cv::multiply(patchGyCh[c], maskFloat, patchGyCh[c]);
        cv::multiply(destGxCh[c], invMaskFloat, destGxCh[c]);
        cv::multiply(destGyCh[c], invMaskFloat, destGyCh[c]);

        const cv::Mat gx = destGxCh[c] + patchGxCh[c];
        const cv::Mat gy = destGyCh[c] + patchGyCh[c];
        cv::Mat lx;
        cv::Mat ly;
        computeLaplacianX(gx, lx);
        computeLaplacianY(gy, ly);
        lapPerChannel[c] = lx + ly;
    }
}

void jacobiChannelCpu(const cv::Mat& lap, const cv::Mat& imgU8, cv::Mat& outU8, int iterations) {
    const int w = imgU8.cols;
    const int h = imgU8.rows;

    cv::Mat u;
    imgU8.convertTo(u, CV_32F);

    const cv::Mat f = lap(cv::Rect(1, 1, w - 2, h - 2));
    cv::Mat a = u.clone();
    cv::Mat b = u.clone();

    for (int iter = 0; iter < iterations; ++iter) {
        const cv::Mat& src = (iter % 2 == 0) ? a : b;
        cv::Mat& dst = (iter % 2 == 0) ? b : a;

        cv::parallel_for_(cv::Range(1, h - 1), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const float* srcAbove = src.ptr<float>(y - 1);
                const float* srcRow = src.ptr<float>(y);
                const float* srcBelow = src.ptr<float>(y + 1);
                const float* fRow = f.ptr<float>(y - 1);
                float* dstRow = dst.ptr<float>(y);

                for (int x = 1; x < w - 1; ++x) {
                    dstRow[x] =
                        0.25f * (srcAbove[x] + srcBelow[x] + srcRow[x - 1] + srcRow[x + 1] - fRow[x - 1]);
                }
            }
        });
    }

    const cv::Mat& result = (iterations % 2 == 0) ? a : b;
    outU8.create(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y) {
        const float* srcRow = result.ptr<float>(y);
        uchar* dstRow = outU8.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            const float value = srcRow[x];
            dstRow[x] = static_cast<uchar>(std::min(255.f, std::max(0.f, value)));
        }
    }
}

#ifdef POISSON_JACOBI_DLOPEN

using cl_int = int;
using cl_uint = unsigned int;
using cl_platform_id = void*;
using cl_device_id = void*;
using cl_context = void*;
using cl_command_queue = void*;
using cl_program = void*;
using cl_kernel = void*;
using cl_mem = void*;

using PFN_clGetPlatformIDs = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using PFN_clGetDeviceIDs = cl_int (*)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
using PFN_clCreateContext = cl_context (*)(const void*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*),
                                           void*, cl_int*);
using PFN_clCreateCommandQueue = cl_command_queue (*)(cl_context, cl_device_id, cl_uint, cl_int*);
using PFN_clCreateProgramWithSource = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using PFN_clBuildProgram = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
using PFN_clCreateKernel = cl_kernel (*)(cl_program, const char*, cl_int*);
using PFN_clCreateBuffer = cl_mem (*)(cl_context, cl_uint, size_t, void*, cl_int*);
using PFN_clEnqueueWriteBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, void*, void**);
using PFN_clEnqueueNDRangeKernel = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint,
                                              void**, void**);
using PFN_clFinish = cl_int (*)(cl_command_queue);
using PFN_clEnqueueReadBuffer = cl_int (*)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, void*, void**);
using PFN_clReleaseMemObject = cl_int (*)(cl_mem);
using PFN_clReleaseKernel = cl_int (*)(cl_kernel);
using PFN_clReleaseProgram = cl_int (*)(cl_program);
using PFN_clReleaseCommandQueue = cl_int (*)(cl_command_queue);
using PFN_clReleaseContext = cl_int (*)(cl_context);
using PFN_clSetKernelArg = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);

struct OpenCLPoisson {
    void* lib = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    bool ok = false;

    PFN_clGetPlatformIDs clGetPlatformIDs = nullptr;
    PFN_clGetDeviceIDs clGetDeviceIDs = nullptr;
    PFN_clCreateContext clCreateContext = nullptr;
    PFN_clCreateCommandQueue clCreateCommandQueue = nullptr;
    PFN_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
    PFN_clBuildProgram clBuildProgram = nullptr;
    PFN_clCreateKernel clCreateKernel = nullptr;
    PFN_clCreateBuffer clCreateBuffer = nullptr;
    PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
    PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
    PFN_clFinish clFinish = nullptr;
    PFN_clEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
    PFN_clReleaseMemObject clReleaseMemObject = nullptr;
    PFN_clReleaseKernel clReleaseKernel = nullptr;
    PFN_clReleaseProgram clReleaseProgram = nullptr;
    PFN_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
    PFN_clReleaseContext clReleaseContext = nullptr;
    PFN_clSetKernelArg clSetKernelArg = nullptr;

    template <typename T>
    bool loadSym(T& fn, const char* name) {
        fn = reinterpret_cast<T>(dlsym(lib, name));
        return fn != nullptr;
    }

    bool init() {
        if (ok) {
            return true;
        }

        const char* libs[] = {"libOpenCL.so", nullptr};
        for (int i = 0; libs[i] != nullptr; ++i) {
            lib = dlopen(libs[i], RTLD_LAZY | RTLD_LOCAL);
            if (lib != nullptr) {
                break;
            }
        }
        if (lib == nullptr) {
            return false;
        }

        if (!loadSym(clGetPlatformIDs, "clGetPlatformIDs")) return false;
        if (!loadSym(clGetDeviceIDs, "clGetDeviceIDs")) return false;
        if (!loadSym(clCreateContext, "clCreateContext")) return false;
        if (!loadSym(clCreateCommandQueue, "clCreateCommandQueue")) return false;
        if (!loadSym(clCreateProgramWithSource, "clCreateProgramWithSource")) return false;
        if (!loadSym(clBuildProgram, "clBuildProgram")) return false;
        if (!loadSym(clCreateKernel, "clCreateKernel")) return false;
        if (!loadSym(clCreateBuffer, "clCreateBuffer")) return false;
        if (!loadSym(clEnqueueWriteBuffer, "clEnqueueWriteBuffer")) return false;
        if (!loadSym(clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel")) return false;
        if (!loadSym(clFinish, "clFinish")) return false;
        if (!loadSym(clEnqueueReadBuffer, "clEnqueueReadBuffer")) return false;
        if (!loadSym(clReleaseMemObject, "clReleaseMemObject")) return false;
        if (!loadSym(clReleaseKernel, "clReleaseKernel")) return false;
        if (!loadSym(clReleaseProgram, "clReleaseProgram")) return false;
        if (!loadSym(clReleaseCommandQueue, "clReleaseCommandQueue")) return false;
        if (!loadSym(clReleaseContext, "clReleaseContext")) return false;
        if (!loadSym(clSetKernelArg, "clSetKernelArg")) return false;

        cl_platform_id platform = nullptr;
        if (clGetPlatformIDs(1, &platform, nullptr) != 0 || platform == nullptr) {
            return false;
        }

        cl_device_id device = nullptr;
        if (clGetDeviceIDs(platform, 1u << 2, 1, &device, nullptr) != 0 || device == nullptr) {
            if (clGetDeviceIDs(platform, 1u << 0, 1, &device, nullptr) != 0 || device == nullptr) {
                return false;
            }
        }

        cl_int err = 0;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != 0 || context == nullptr) {
            return false;
        }

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != 0 || queue == nullptr) {
            return false;
        }

        static const char* kKernel = R"CLC(
__kernel void jacobi(__global const float* src, __global float* dst,
                     __global const float* f, int width, int height) {
    const int x = (int)get_global_id(0) + 1;
    const int y = (int)get_global_id(1) + 1;
    if (x >= width - 1 || y >= height - 1) return;
    const int idx = y * width + x;
    const float sum = src[idx - width] + src[idx + width] + src[idx - 1] + src[idx + 1];
    dst[idx] = 0.25f * (sum - f[(y - 1) * (width - 2) + (x - 1)]);
}
)CLC";

        program = clCreateProgramWithSource(context, 1, &kKernel, nullptr, &err);
        if (err != 0 || program == nullptr) {
            return false;
        }

        if (clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr) != 0) {
            return false;
        }

        kernel = clCreateKernel(program, "jacobi", &err);
        ok = (err == 0 && kernel != nullptr);
        return ok;
    }

    bool solveChannel(const cv::Mat& lap, const cv::Mat& imgU8, cv::Mat& outU8, int iterations) {
        if (!init()) {
            return false;
        }

        const int w = imgU8.cols;
        const int h = imgU8.rows;
        const size_t plane = static_cast<size_t>(w) * static_cast<size_t>(h);
        const size_t inner = static_cast<size_t>(w - 2) * static_cast<size_t>(h - 2);

        cv::Mat u;
        imgU8.convertTo(u, CV_32F);
        const cv::Mat f = lap(cv::Rect(1, 1, w - 2, h - 2));

        std::vector<float> bufA(u.begin<float>(), u.end<float>());
        std::vector<float> outHost(plane);

        const cl_uint memReadWrite = 1u << 0;
        const cl_uint memReadOnly = 1u << 2;
        cl_int err = 0;
        cl_mem memA = clCreateBuffer(context, memReadWrite, plane * sizeof(float), nullptr, &err);
        cl_mem memB = clCreateBuffer(context, memReadWrite, plane * sizeof(float), nullptr, &err);
        cl_mem memF = clCreateBuffer(context, memReadOnly, inner * sizeof(float), nullptr, &err);
        if (memA == nullptr || memB == nullptr || memF == nullptr) {
            if (memA) clReleaseMemObject(memA);
            if (memB) clReleaseMemObject(memB);
            if (memF) clReleaseMemObject(memF);
            return false;
        }

        clEnqueueWriteBuffer(queue, memA, 1, 0, plane * sizeof(float), bufA.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, memF, 1, 0, inner * sizeof(float), f.ptr<float>(), 0, nullptr, nullptr);

        const int argW = w;
        const int argH = h;
        for (int iter = 0; iter < iterations; ++iter) {
            cl_mem src = (iter % 2 == 0) ? memA : memB;
            cl_mem dst = (iter % 2 == 0) ? memB : memA;

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &memF);
            clSetKernelArg(kernel, 3, sizeof(int), &argW);
            clSetKernelArg(kernel, 4, sizeof(int), &argH);

            const size_t gw = static_cast<size_t>(std::max(1, w - 2));
            const size_t gh = static_cast<size_t>(std::max(1, h - 2));
            const size_t global[2] = {gw, gh};
            clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        }

        clFinish(queue);
        const cl_mem result = (iterations % 2 == 0) ? memA : memB;
        clEnqueueReadBuffer(queue, result, 1, 0, plane * sizeof(float), outHost.data(), 0, nullptr, nullptr);
        clFinish(queue);

        outU8.create(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            uchar* row = outU8.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                const float value = outHost[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)];
                row[x] = static_cast<uchar>(std::min(255.f, std::max(0.f, value)));
            }
        }

        clReleaseMemObject(memA);
        clReleaseMemObject(memB);
        clReleaseMemObject(memF);
        return true;
    }
};

OpenCLPoisson gOpenCL;

#endif  // POISSON_JACOBI_DLOPEN

bool runJacobiPoissonCore(const BenchCase& bench, cv::Mat& output, int iterations, bool useOpenCL) {
    SeamlessRoi roi;
    if (!extractSeamlessRoi(bench, roi)) {
        return false;
    }

    cv::Mat mask = roi.maskROI.clone();
    std::vector<cv::Mat> lapPerChannel;
    buildNormalCloneLaplacian(roi.dstROI, roi.srcROI, mask, lapPerChannel);

    std::vector<cv::Mat> channels;
    cv::split(roi.dstROI, channels);
    for (int c = 0; c < 3; ++c) {
        cv::Mat solved;
        bool done = false;
#ifdef POISSON_JACOBI_DLOPEN
        if (useOpenCL) {
            done = gOpenCL.solveChannel(lapPerChannel[c], channels[c], solved, iterations);
        }
#else
        (void)useOpenCL;
#endif
        if (!done) {
            jacobiChannelCpu(lapPerChannel[c], channels[c], solved, iterations);
        }
        channels[c] = solved;
    }

    cv::merge(channels, roi.dstROI);
    bench.dst.copyTo(output);
    roi.dstROI.copyTo(output(roi.roi_d));
    return true;
}

}  // namespace

bool isOpenCLPoissonAvailable() {
#ifdef POISSON_JACOBI_DLOPEN
    return gOpenCL.init();
#else
    return false;
#endif
}

bool runJacobiPoissonClone(const BenchCase& bench, cv::Mat& output, int iterations, bool useOpenCL) {
    if (useOpenCL && !isOpenCLPoissonAvailable()) {
        return false;
    }
    return runJacobiPoissonCore(bench, output, iterations, useOpenCL);
}
