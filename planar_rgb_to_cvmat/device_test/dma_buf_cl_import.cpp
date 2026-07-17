#include "dma_buf_cl_import.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef OCR_OPENCL_DLOPEN
#include "opencl_dynload.h"
#endif

namespace {

using clGetExtPlat_fn = void*(CL_API_CALL*)(cl_platform_id, const char*);
using clGetExt_fn = void*(CL_API_CALL*)(const char*);

clImportMemoryARM_fn resolveImport(cl_platform_id platform) {
#ifdef OCR_OPENCL_DLOPEN
    if (clGetExtensionFunctionAddressForPlatform_dyn) {
        auto* fn = reinterpret_cast<clGetExtPlat_fn>(clGetExtensionFunctionAddressForPlatform_dyn);
        return reinterpret_cast<clImportMemoryARM_fn>(fn(platform, "clImportMemoryARM"));
    }
    if (clGetExtensionFunctionAddress_dyn) {
        auto* fn = reinterpret_cast<clGetExt_fn>(clGetExtensionFunctionAddress_dyn);
        return reinterpret_cast<clImportMemoryARM_fn>(fn("clImportMemoryARM"));
    }
    return nullptr;
#else
    (void)platform;
    return clImportMemoryARM;
#endif
}

}  // namespace

bool clHasArmImportMemory(cl_platform_id platform) {
    size_t n = 0;
    if (clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, nullptr, &n) != CL_SUCCESS ||
        n == 0) {
        return false;
    }
    std::vector<char> buf(n + 1, 0);
    if (clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, n, buf.data(), nullptr) != CL_SUCCESS) {
        return false;
    }
    return std::strstr(buf.data(), "cl_arm_import_memory") != nullptr;
}

cl_mem clImportDmaBuf(cl_context ctx, cl_platform_id platform, cl_mem_flags flags, int dma_buf_fd,
                      size_t size, cl_int* err_out, DmaBufImportStatus* status) {
    DmaBufImportStatus local{};
    DmaBufImportStatus* st = status ? status : &local;

    if (dma_buf_fd < 0) {
        st->detail = "invalid dma_buf fd";
        if (err_out) {
            *err_out = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    st->ext_present = clHasArmImportMemory(platform);
    if (!st->ext_present) {
        st->detail = "cl_arm_import_memory not advertised on this platform";
        if (err_out) {
            *err_out = CL_INVALID_OPERATION;
        }
        return nullptr;
    }

    clImportMemoryARM_fn import_fn = resolveImport(platform);
    if (!import_fn) {
        st->detail = "clImportMemoryARM symbol not found (dlsym / ICD)";
        if (err_out) {
            *err_out = CL_INVALID_OPERATION;
        }
        return nullptr;
    }

    // properties: type = DMA_BUF, then terminator
    const cl_import_properties_arm props[] = {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_DMA_BUF_ARM,
        0,
    };

    cl_int err = CL_SUCCESS;
    // memory argument is the fd cast to void* per ARM extension
    cl_mem mem =
        import_fn(ctx, flags, props, reinterpret_cast<void*>(static_cast<intptr_t>(dma_buf_fd)),
                  size, &err);
    if (err_out) {
        *err_out = err;
    }
    if (err != CL_SUCCESS || !mem) {
        st->detail = "clImportMemoryARM failed err=" + std::to_string(err);
        return nullptr;
    }
    st->import_ok = true;
    st->detail = "imported dma_buf fd via clImportMemoryARM";
    return mem;
}

const char* dmaBufImportHint() {
    return "ION/DMA-BUF -> clImportMemoryARM(CL_IMPORT_TYPE_DMA_BUF_ARM) -> cl_mem -> "
           "planar_rgb_to_cv32fc3_ppx (no H2D). Keep dst in another DMA-BUF if next stage is GPU.";
}
