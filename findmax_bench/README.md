# findMax / enhanceBrightness OpenCL bench

## Pipeline (`run_enhance.ps1`)

| path | what |
|---|---|
| **A baseline** | `findmax_orig_2d` + `enhance_brightness`（2 kernels, 2D 1px） |
| **B fused** | `findMaxAndEnhance` **1 kernel, 1 workgroup**（只有 local barrier；跨 WG spin 会 CL_-14） |
| **C opt** | `findmax_opt` + `enhance_brightness_opt`（2 kernels, stride/`half4`，通常最快） |

Enhance: `divisor=fmin(1,max)`, `src *= 1/divisor`.

```powershell
cd findmax_bench
.\scripts\build_ohos.ps1
.\scripts\run_enhance.ps1
```

## findmax only

`.\scripts\run_device.ps1`
