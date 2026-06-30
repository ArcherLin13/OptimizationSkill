# seamless_clone_fft

OpenCV 4.9 `NORMAL_CLONE` reimplementation (DST/DFT). Target: `maxDiff=0` vs `cv::seamlessClone`.

**Stable baseline (~100ms):** git tag `seamless-fft-100ms-pass`

## Production usage

```cpp
#include "seamless_clone_fft.h"

seamless_clone_fft::Context ctx;
ctx.seamlessClone(src, dst, mask, center, output);
```

## Optimizations

- Ported OpenCV 4.9 `dxt.cpp` float complex 1D DFT (`ocv_dft_32f.cpp`) with per-length plan cache
- ARM NEON radix-4 butterfly (`ocv_dft_neon_r4.h`) on aarch64
- 3 RGB channels parallel Poisson DST + `cv::setNumThreads(n/3)`
- Context buffer reuse, NEON DST prep / eigen / clamp

## Files

| File | Role |
|------|------|
| `seamless_clone_fft.h/cpp` | Clone API |
| `sc_fft_rows.cpp` | Row `DFT_ROWS` wrapper |
| `ocv_dft_32f.cpp` | OpenCV-compatible 1D complex FFT |
| `ocv_dft_neon_r4.h` | NEON radix-4 |

Link OpenCV **core + imgproc** only.
