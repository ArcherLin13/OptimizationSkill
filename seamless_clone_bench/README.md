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
