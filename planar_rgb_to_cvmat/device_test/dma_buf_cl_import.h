#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <cstddef>
#include <string>

// Import an ION / DMA-BUF fd into OpenCL via cl_arm_import_memory (Mali / HiSilicon).
// Returns nullptr if extension missing or import fails; check err_out.
//
// Typical camera / ISP path:
//   ion_fd  -> clImportDmaBuf(...) -> cl_mem  -> planar_rgb_to_cv32fc3_ppx
// Output can also be an imported DMA-BUF if the next consumer is GPU/ISP.

struct DmaBufImportStatus {
    bool ext_present = false;
    bool import_ok = false;
    std::string detail;
};

// Probe platform for "cl_arm_import_memory". Call after opencl_load().
bool clHasArmImportMemory(cl_platform_id platform);

// Import dma_buf fd. size = byte size of the allocation (or CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM).
// flags: CL_MEM_READ_ONLY / CL_MEM_WRITE_ONLY / CL_MEM_READ_WRITE
cl_mem clImportDmaBuf(cl_context ctx, cl_platform_id platform, cl_mem_flags flags, int dma_buf_fd,
                      size_t size, cl_int* err_out, DmaBufImportStatus* status = nullptr);

// Human-readable note for logs.
const char* dmaBufImportHint();
