# findMaxValue / enhanceBrightness OpenCL bench

## findmax only

`orig` vs `base_1px` vs `opt_stride` — see `.\scripts\run_device.ps1`.

## findmax + enhanceBrightness

| path | kernels |
|---|---|
| **baseline** | `findmax_orig_2d` + `enhance_brightness`（2 次 enqueue） |
| **fused** | `findMaxAndEnhance`：基于 opt 的 grid-stride/`half4`，phase1 max + grid sync + phase2 enhance（1 次 enqueue） |

Enhance: `divisor = fmin(1, max_value)`，`src *= 1/divisor`（in-place）。

```powershell
cd findmax_bench
.\scripts\build_ohos.ps1
.\scripts\run_enhance.ps1
```

输出会校验整图 vs CPU，并报 baseline（findmax+enhance 分段/合计）与 fused 的 kernel 耗时。
