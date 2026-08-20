# f32 -> f16 OpenCL microbench

Standalone GPU benchmark for float32 to float16 conversion (not part of findmax).

## Kernels

| Case | File | Launch |
|------|------|--------|
| `1d_n` | `f32_to_f16_1d_n.cl` | `gws ~= n`, 1 float/WI |
| `2d_wh` | `f32_to_f16_2d.cl` | `gws ~= (W,H)` |
| `stride_f4` | `f32_to_f16_stride.cl` | fixed `gws = lws*nwg`, float4 grid-stride |

## HarmonyOS device

```powershell
cd f32_to_f16_bench
.\scripts\build_ohos.ps1
.\scripts\run_device.ps1
```

Device path: `/data/local/tmp/f32_to_f16`

## Linux host (optional)

Requires `g++`/`clang++`, CMake, and system OpenCL (`libOpenCL` + GPU driver).

```bash
cd f32_to_f16_bench
mkdir -p build/linux && cd build/linux
cmake ../.. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target ocl_test_f32_to_f16
cd ../..
./build/linux/ocl_test_f32_to_f16 --stride f32_to_f16_stride.cl --width 5760 --height 4320
```

Run from `f32_to_f16_bench/` so default `.cl` paths resolve.
