# findMaxValue OpenCL bench

Compare **orig (2D)** vs **mine (1D)** on a `half` image (`5760×4320` default).

Same logic for both:

1. Each WI loads one pixel
2. Workgroup local reduction → local max
3. `lid==0` atomically updates global max

| | orig | mine |
|---|---|---|
| NDRange | 2D (`lws` 16×16) | 1D (`lws` 256) |
| local mem | `float s_max[256]` | `half lmax[256]` |

Orig matches the original structure; only crash hazards removed (no early-return before `barrier`, OOB guard, `local_size<=256`).

## Build / run (HarmonyOS)

```powershell
cd findmax_bench
.\scripts\build_ohos.ps1
.\scripts\run_device.ps1
```

Reports **kernel-only** time for both, plus `orig/mine` speedup.
