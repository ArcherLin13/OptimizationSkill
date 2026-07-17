# Planar RGB → `cv::Mat` (`CV_32FC3`) OpenCL / NEON

Converts **planar** RGB (R plane | G plane | B plane) into OpenCV **interleaved** `CV_32FC3` (BGR float).

## ION / DMA-BUF：可以用 GPU，而且应该用

如果你的图已经在 **ION / DMA-BUF** 里，GPU 可以直接 `clImportMemoryARM(DMA_BUF)` 绑成 `cl_mem`，**没有 H2D 拷贝**。这种场景下：

| 做法 | 评价 |
|------|------|
| GPU kernel 转 `CV_32FC3`（dst 也在 ION） | **正确路径** — 下一环还是 GPU/ISP 时更合适 |
| map ION → CPU NEON → 再给 GPU | 多一次 CPU 触碰 + cache 同步，通常更差 |
| 和 NEON 比 “谁更快” | 都是打同一块 DRAM；GPU 时间接近 NEON 很正常 |

之前说 “prefer NEON” 是针对：**数据在普通 host 堆、还要 `clEnqueueWrite/Read`** 的 bench。那不是你的 ION 流水线。

```text
camera/ISP ION (planar RGB)
        │  clImportDmaBuf(fd)     ← zero-copy
        ▼
   cl_mem src
        │  planar_rgb_to_cv32fc3_ppx
        ▼
   cl_mem dst (also ION / DMA-BUF)
        │
        ▼
   next GPU / NPU / display stage
```

Helper: `device_test/dma_buf_cl_import.h`

```cpp
#include "dma_buf_cl_import.h"

cl_int err = 0;
cl_mem src = clImportDmaBuf(ctx, plat, CL_MEM_READ_ONLY, ion_fd_src, nbytes, &err);
cl_mem dst = clImportDmaBuf(ctx, plat, CL_MEM_WRITE_ONLY, ion_fd_dst, nbytes_out, &err);
// then set kernel args to src/dst — no WriteBuffer / ReadBuffer
```

设备上会打印：`DMA-BUF import (cl_arm_import_memory): YES/no`。

## Layout

| | Memory |
|---|--------|
| **Input** | `R[0..N)`, `G[0..N)`, `B[0..N)` contiguous planes |
| **Output** | OpenCV `CV_32FC3`: per pixel `[B, G, R]` float |

## ROI crop（替代 `rgb_mat(roi)`）

整图是 planar buffer 时，不要先整图 `trans2cv`，对每个 text box：

```cpp
#include "device_test/planar_roi_crop.h"

// old: cv::Mat crop_aabb = rgb_mat(roi);
// new:
cv::Mat crop_aabb = planar::cropAabbFromBuffers(r, g, b, width, height, width, roi);
// crop_aabb: CV_32FC3 BGR continuous — 可直接 warpPerspective
```

Packed `[R|G|B]`：`planar::cropAabbFromPlanarPackedF32(planar, w, h, stride, roi)`。

### 怎么测（HarmonyOS）— 默认就是你的场景

```powershell
cd planar_rgb_to_cvmat
.\scripts\build_ohos.ps1
.\scripts\run_roi_crop.ps1
# 默认: 4096x3072, 16 boxes of 1000x150
```

自定义：

```powershell
.\scripts\run_roi_crop.ps1 -Width 4096 -Height 3072 -Boxes 16 -BoxW 1000 -BoxH 150 -Runs 30
```

输出会对比：
- 正确性：ROI crop == 整图 trans 再 crop（等价 `rgb_mat(roi)`）
- 耗时：整图 trans 一次 vs 16 次 ROI trans 合计

## NEON（仅当必须落到 host 整图 Mat 时）

```cpp
#include "device_test/neon_planar_to_cv32fc3.h"
neon_planar_rgb_f32_to_cv32fc3(r, g, b, (float*)mat.data, w, h, w, w * 3);
neon_planar_rgb_f32_to_cv32fc3_mt(r, g, b, dst, w, h, w, w * 3, 0);
```

## HarmonyOS device test

```powershell
cd planar_rgb_to_cvmat
.\scripts\build_ohos.ps1
.\scripts\run_device.ps1
```

`GPU *` 行 = **设备常驻**（等价于 ION 已 import 后的 kernel 时间）。

## Kernels

| Kernel | Use |
|--------|-----|
| `planar_rgb_to_cv32fc3_ppx` | 推荐；`-DPIXELS_PER_WI=8` |
| `planar_rgb_packed_to_cv32fc3_ppx` | 单 buffer `[R\|G\|B]` |
