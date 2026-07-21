# findMaxValue OpenCL bench

GPU kernel: find max over a `half` image (`5760×4320` default).

Baseline algorithm:

1. Each work-item loads one pixel
2. Workgroup local reduction → local max
3. Leader updates global max (`atomic_max` via cmpxchg on half bits)

## Build / run (HarmonyOS)

```powershell
cd findmax_bench
.\scripts\build_ohos.ps1
.\scripts\run_device.ps1
```

Reports **kernel-only** time (`CL_QUEUE_PROFILING_ENABLE`), plus correctness vs CPU reference max.
