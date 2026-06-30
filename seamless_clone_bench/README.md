# seamlessClone OHOS Benchmark

在华为 HarmonyOS 手机上对比 `cv::seamlessClone(NORMAL_CLONE)` 基线与优化路径的**耗时**和**正确性**。

## 测试内容

| 路径 | 说明 | 正确性要求 |
|------|------|-----------|
| `baseline` | OpenCV `seamlessClone` | 参考标准 |
| `pooled_reuse` | 预分配 buffer，避免重复分配 | 与 baseline **完全一致** |
| `aligned_736x128` | pad 到 32 对齐 (729×126 → 736×128) | PSNR ≥ 42dB, maxDiff ≤ 6 |
| `half_res` | 半分辨率 solve + 上采样 | 近似路径，单独报告 |

测试尺寸默认 **729×126**（与你当前场景一致），mask 为椭圆区域。

## 前置条件

1. 已安装 **DevEco Studio**（本机路径示例）：
   `C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native`
2. 手机开启开发者模式，USB 连接，`hdc list targets` 能看到设备
3. 需要先用脚本交叉编译 **OpenCV**（仅 core/imgproc/photo）

## 预编译可执行文件（OHOS arm64）

若不想本地编译，可直接使用仓库内预编译二进制：

```text
bin/arm64-v8a/seamless_clone_bench
```

推送到手机时仍需本机 OpenCV `.so` 与 `libc++_shared.so`（见 `deploy.ps1`），**预编译包不含 OpenCV 库**。

## 另一台电脑：配置 DevEco 工具链路径

DevEco 装在不同盘符/目录时，任选一种方式：

### 方式 1：本地配置文件（推荐）

```powershell
copy scripts\config.local.ps1.example scripts\config.local.ps1
notepad scripts\config.local.ps1
```

修改 `$OHOS_NATIVE` 为你的 `openharmony\native` 目录，例如：

```powershell
$OHOS_NATIVE = "D:\Huawei\DevEco Studio\sdk\default\openharmony\native"
```

`config.local.ps1` 不会提交到 Git。

### 方式 2：命令行参数

```powershell
.\scripts\build.ps1 -OhosNative "D:\你的路径\openharmony\native"
.\scripts\deploy_chip_prod.ps1 -OhosNative "D:\你的路径\openharmony\native"
```

### 方式 3：环境变量

```powershell
$env:OHOS_NATIVE = "D:\你的路径\openharmony\native"
.\scripts\build.ps1
```

`$OHOS_NATIVE` 必须能访问到：

```text
build\cmake\ohos.toolchain.cmake
build-tools\cmake\bin\cmake.exe
llvm\bin\clang++.exe
..\toolchains\hdc.exe
```

## 用本机 OpenCV（推荐，匹配 /chip_prod/lib64）

从手机拷贝 4 个 so 到 `opencv/`：

- `libopencv_core.so*`
- `libopencv_imgcodecs.so*`
- `libopencv_imgproc.so*`
- `libopencv_photo.so*`

头文件放到 `opencv/include/opencv4/`（版本需与 so 一致）。

```powershell
.\scripts\build.ps1
.\scripts\deploy_chip_prod.ps1
```

设备目录（默认）：

```text
/data/vendor/camera/
  seamless_clone_bench    # 可执行文件
  out/                    # 对比图输出（grid_results.bmp 等）
  case/                   # push_case 推送的测试图（可选）
```

在设备上手动运行（图片写入 `./out/`）：

```bash
cd /data/vendor/camera
export LD_LIBRARY_PATH=/chip_prod/lib64:$LD_LIBRARY_PATH
./seamless_clone_bench --case text
ls out/
```

拉回 PC：`.\scripts\pull_results.ps1`（默认拉 `/data/vendor/camera/out`）

## 在业务代码里替换 seamlessClone

拷贝整个文件夹到你的工程：

```text
seamless_clone_bench/lib/seamless_clone_jacobi/
  seamless_clone_jacobi.h
  seamless_roi.h
  seamless_clone_jacobi.cpp
  README.md
```

```cpp
#include "seamless_clone_jacobi.h"
seamless_clone_jacobi::seamlessClone(src, dst, mask, center, output);
```

仅依赖 OpenCV **core + imgproc**，无 OpenCL、无 `opencv_photo`、无 `dl`。

### Phase A：FFT 完全一致验证（进行中）

`lib/seamless_clone_fft/` 按 OpenCV 4.9 源码移植了 DST/DFT Poisson 求解器。  
设备上跑 benchmark 看 **`poisson_fft`** 行：必须 `maxDiff=0` 且标记为 `SAME` 才算 Phase A 通过；通过后再做 FFT 加速（Phase B）。

默认 **不再** 需要 `build_opencv.ps1`。若要用自编的 OpenCV 4.9：

```powershell
.\scripts\build.ps1 -UseBundledOpenCV
```

## 构建步骤（Windows PowerShell）

```powershell
cd d:\WorkStation\OptimizationSkill\seamless_clone_bench

# 1) 交叉编译 OpenCV（首次较慢，约 10~30 分钟）
.\scripts\build_opencv.ps1

# 2) 编译 benchmark
.\scripts\build.ps1

# 3) 推送到手机并运行
.\scripts\deploy.ps1
```

## 自定义 OpenCV 路径

若你已有 HarmonyOS 版 OpenCV：

```powershell
.\scripts\build.ps1 -OpenCvOhosDir "D:\path\to\opencv-ohos-arm64"
.\scripts\deploy.ps1 -OpenCvOhosDir "D:\path\to\opencv-ohos-arm64"
```

## 输出示例

```text
baseline: avg=280.00ms ...
pooled_reuse: avg=275.00ms ...
aligned_736x128: avg=95.00ms ...
half_res: avg=70.00ms ...
pooled_reuse correctness: ... PASS (must be identical)
aligned_736x128 correctness: ... PASS (FFT-friendly pad)
```

## 工具链

使用 DevEco 自带：

- `ohos.toolchain.cmake`
- `cmake.exe` / `ninja.exe`
- `hdc.exe` 部署

## 目录结构

```text
seamless_clone_bench/
  CMakeLists.txt
  src/
  scripts/
    build_opencv.ps1
    build.ps1
    deploy.ps1
  third_party/          # build_opencv 后生成
  build/                # 编译输出
```
