# Planar RGB → `cv::Mat` (`CV_32FC3`) OpenCL kernel

Converts **planar** RGB (full R plane, then G, then B) into OpenCV **interleaved** `CV_32FC3` (BGR float by default).

## Layout

| | Memory |
|---|--------|
| **Input** | `R[0..N)`, `G[0..N)`, `B[0..N)` contiguous planes (`N = height * stride`) |
| **Output** | OpenCV `CV_32FC3`: per pixel `[B, G, R]` float |

## HarmonyOS device test

```powershell
cd planar_rgb_to_cvmat
.\scripts\build_ohos.ps1
.\scripts\run_device.ps1
```

Optional size / runs:

```powershell
.\scripts\run_device.ps1 -Width 1280 -Height 720 -Runs 30
```

Default remote path: `/data/local/tmp/planar_rgb`

Expected output:

```text
Device: ...
=== results (GPU vs CPU ref, OpenCV BGR float) ===
  float separate planes         max_diff=...  kernel=... ms  OK
  float packed [R|G|B]          max_diff=...  kernel=... ms  OK
  uchar separate (/255)         max_diff=...  kernel=... ms  OK
```

## Kernels

| Kernel | Use when |
|--------|----------|
| `planar_rgb_to_cv32fc3` | Separate `r`, `g`, `b` buffers |
| `planar_rgb_packed_to_cv32fc3` | One buffer: `[R\|G\|B]` |

## Compile flags

| Flag | Effect |
|------|--------|
| *(none)* | `float` planes → float BGR |
| `-DINPUT_UCHAR` | `uchar` planes, scale `/255` |
| `-DOUT_RGB` | write RGB instead of OpenCV BGR |
