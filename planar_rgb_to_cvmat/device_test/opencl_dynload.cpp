#include "opencl_dynload.h"

#include <dlfcn.h>
#include <cstdio>

#define LOAD(name)                                                                                 \
    do {                                                                                           \
        name##_dyn = reinterpret_cast<decltype(name##_dyn)>(dlsym(handle, #name));                 \
        if (!(name##_dyn)) {                                                                       \
            std::fprintf(stderr, "dlsym failed: %s (%s)\n", #name, dlerror());                     \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

CL_DYN(cl_int) (*clGetPlatformIDs_dyn)(cl_uint, cl_platform_id*, cl_uint*);
CL_DYN(cl_int) (*clGetDeviceIDs_dyn)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
CL_DYN(cl_int) (*clGetDeviceInfo_dyn)(cl_device_id, cl_device_info, size_t, void*, size_t*);
CL_DYN(cl_context) (*clCreateContext_dyn)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                          void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*,
                                          cl_int*);
CL_DYN(cl_command_queue) (*clCreateCommandQueue_dyn)(cl_context, cl_device_id, cl_command_queue_properties,
                                                     cl_int*);
CL_DYN(cl_program) (*clCreateProgramWithSource_dyn)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
CL_DYN(cl_int) (*clBuildProgram_dyn)(cl_program, cl_uint, const cl_device_id*, const char*,
                                     void(CL_CALLBACK*)(cl_program, void*), void*);
CL_DYN(cl_int) (*clGetProgramBuildInfo_dyn)(cl_program, cl_device_id, cl_program_build_info, size_t, void*,
                                            size_t*);
CL_DYN(cl_kernel) (*clCreateKernel_dyn)(cl_program, const char*, cl_int*);
CL_DYN(cl_int) (*clSetKernelArg_dyn)(cl_kernel, cl_uint, size_t, const void*);
CL_DYN(cl_mem) (*clCreateBuffer_dyn)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
CL_DYN(cl_int) (*clEnqueueWriteBuffer_dyn)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint,
                                           const cl_event*, cl_event*);
CL_DYN(cl_int) (*clEnqueueNDRangeKernel_dyn)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                              const size_t*, cl_uint, const cl_event*, cl_event*);
CL_DYN(cl_int) (*clEnqueueReadBuffer_dyn)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint,
                                          const cl_event*, cl_event*);
CL_DYN(cl_int) (*clFinish_dyn)(cl_command_queue);
CL_DYN(cl_int) (*clWaitForEvents_dyn)(cl_uint, const cl_event*);
CL_DYN(cl_int) (*clGetEventProfilingInfo_dyn)(cl_event, cl_profiling_info, size_t, void*, size_t*);
CL_DYN(cl_int) (*clReleaseMemObject_dyn)(cl_mem);
CL_DYN(cl_int) (*clReleaseKernel_dyn)(cl_kernel);
CL_DYN(cl_int) (*clReleaseProgram_dyn)(cl_program);
CL_DYN(cl_int) (*clReleaseCommandQueue_dyn)(cl_command_queue);
CL_DYN(cl_int) (*clReleaseContext_dyn)(cl_context);
CL_DYN(cl_int) (*clReleaseEvent_dyn)(cl_event);

bool opencl_load() {
    static const char* kPaths[] = {
        "libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib/libOpenCL.so",
        "/system/lib/libOpenCL.so",
        nullptr,
    };

    void* handle = nullptr;
    for (int i = 0; kPaths[i]; ++i) {
        handle = dlopen(kPaths[i], RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            std::fprintf(stderr, "OpenCL loaded: %s\n", kPaths[i]);
            break;
        }
    }
    if (!handle) {
        std::fprintf(stderr, "dlopen libOpenCL.so failed: %s\n", dlerror());
        return false;
    }

    LOAD(clGetPlatformIDs);
    LOAD(clGetDeviceIDs);
    LOAD(clGetDeviceInfo);
    LOAD(clCreateContext);
    LOAD(clCreateCommandQueue);
    LOAD(clCreateProgramWithSource);
    LOAD(clBuildProgram);
    LOAD(clGetProgramBuildInfo);
    LOAD(clCreateKernel);
    LOAD(clSetKernelArg);
    LOAD(clCreateBuffer);
    LOAD(clEnqueueWriteBuffer);
    LOAD(clEnqueueNDRangeKernel);
    LOAD(clEnqueueReadBuffer);
    LOAD(clFinish);
    LOAD(clWaitForEvents);
    LOAD(clGetEventProfilingInfo);
    LOAD(clReleaseMemObject);
    LOAD(clReleaseKernel);
    LOAD(clReleaseProgram);
    LOAD(clReleaseCommandQueue);
    LOAD(clReleaseContext);
    LOAD(clReleaseEvent);
    return true;
}
