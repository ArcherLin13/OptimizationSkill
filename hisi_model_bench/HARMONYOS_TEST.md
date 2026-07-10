# HiSilicon 小模型：加载 vs 推理 能否并行

基于 HarmonyOS **MindSpore Lite**（`libmindspore_lite_ndk.so`），在海思 NPU（NNRT）或 CPU 上测试。

## 测什么

| 场景 | 含义 |
|------|------|
| **serial** | 先 `load` 模型 B，再 `infer` 模型 A（串行） |
| **overlap** | **线程 1** 加载 B，**主线程** 同时推理 A |
| **EnableParallel** | MindSpore 内部算子并行开/关（只影响推理） |

若 `overlap wall ≈ max(load, infer)` → 加载与执行**能并行**  
若 `overlap wall ≈ load + infer` → 设备上**串行**（NPU 被占满）

## 1. 准备模型

任选其一：

```powershell
# A. 生成极小测试模型（PC 需 pip install mindspore）
python generate_tiny_model.py

# B. 复制你自己的小模型
copy your_model.ms testdata\tiny.ms
```

## 2. 编译

```powershell
.\scripts\build_ohos.ps1
```

输出：`build\ohos-arm64\ms_bench`

## 3. 推到手机跑

```powershell
.\scripts\run_device.ps1
# 指定自己的模型：
.\scripts\run_device.ps1 -ModelPath D:\path\to\your.ms
# 只用 CPU：
.\scripts\run_device.ps1 -Device cpu
```

## 4. 输出示例

```text
serial:  load_B=120 ms + infer_A=2 ms  => total=122 ms
overlap: wall=95 ms  (load_B=120 ms, infer_A=2 ms parallel)
overlap saves ~27 ms vs serial (1.28x)

parallel=false: 2.100 ms
parallel=true:  1.950 ms
```

## 5. 直接回答你的问题

**加载和执行能不能并行？**

- **应用层**：可以用多线程一边 `ModelBuildFromFile` 一边 `ModelPredict`（overlap 测试）
- **硬件层**：海思 NPU 可能仍排队，需看 `overlap wall` 是否小于 `load+infer`
- **框架层**：`OH_AI_ContextSetEnableParallel` 是**同一模型推理内部**并行，不是 load+infer 并行

## 依赖

- DevEco OHOS NDK（已含 MindSpore Lite 头文件和 `libmindspore_lite_ndk.so`）
- 手机上有 `libmindspore_lite_ndk.so`（系统自带）
- `.ms` 模型文件（MindIR 格式）
