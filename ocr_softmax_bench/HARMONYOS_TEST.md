# HarmonyOS 2D softmax 设备测试

## 1. 准备测试数据（PC）

```powershell
cd ocr_softmax_bench
node generate_testdata.js
```

## 2. 编译 ocl_test_2d（OHOS arm64）

OpenCL 头文件已包含在仓库 `third_party/OpenCL-Headers/`，无需额外下载。

```powershell
.\scripts\build_ohos_ocl_test.ps1
```

成功输出：`build\ohos-ocl-test\ocl_test_2d`

若链接报 `-lOpenCL` 找不到，两种方式：

**A. 推荐：从手机拉 libOpenCL.so 再编**

```powershell
# 手机 USB 连上，hdc 可用
.\scripts\pull_opencl_from_device.ps1
.\scripts\build_ohos_ocl_test.ps1 -OpenCLLibrary third_party\ohos\libOpenCL.so
```

**B. 允许运行时加载（已默认开启 `--allow-shlib-undefined`）**

NDK 无 libOpenCL 也可链接；真机运行时会加载 `/vendor/lib64/libOpenCL.so`。

## 3. 推到手机并跑

```powershell
.\scripts\run_device_2d.ps1
```

或手动：

```powershell
hdc file send build\ohos-ocl-test\ocl_test_2d /data/vendor/camera/
hdc file send softmax_ocr_opt_2d.cl /data/vendor/camera/
hdc file send testdata\logits.bin /data/vendor/camera/testdata/
hdc file send testdata\probs_ref.bin /data/vendor/camera/testdata/
hdc shell "cd /data/vendor/camera && chmod +x ocl_test_2d && ./ocl_test_2d --data testdata --local-char 512 --runs 20"
```

## 4. 输出说明

```text
correctness max|diff|=...  rows_bad=0  PASS
OpenCL profiling (kernel only): avg=XX.XXX ms
```

- **正确性**：对比 `probs_ref.bin`，`max|diff| < 1e-3`
- **Profiling**：OpenCL event 时间（纯 kernel，不含 readback）

## 5. Launch 参数（2D kernel）

| 项 | 值 |
|----|-----|
| kernel | `softmax_ocr_opt_2d` |
| build | `-DLOCAL_CHAR=512` |
| global | `{128, 512}` |
| local | `{1, 512}` |
| arg4 reduce_buf | local 2048B，`nullptr` |

512 失败改 `--local-char 256`。
