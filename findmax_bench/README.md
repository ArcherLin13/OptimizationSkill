# findMax / enhanceBrightness OpenCL bench

## Pipeline (`run_enhance.ps1`)

| path | what |
|---|---|
| **A baseline** | `findmax_orig_2d` + `enhance_brightness`（2 kernels） |
| **B fused** | `findMaxAndEnhance` **1 kernel / 1 WG**（local barrier 当“等齐”） |
| **C opt** | `findmax_opt` + `enhance_opt`（2 kernels 多 WG；**host 队列顺序 = 全局同步**，推荐） |

在这台 HarmonyOS GPU 上，**单 kernel 多 WG + device 自旋等齐会挂**（试过全 lane / 仅 lid0，均 CL hang/-14）。  
多 WG 的“先 max 再 enhance”请用 **C：两个 kernel 连续 enqueue**，语义等价 fuse，但不在 device 里空转等待。

```powershell
cd findmax_bench
.\scripts\build_ohos.ps1
.\scripts\run_enhance.ps1
```
