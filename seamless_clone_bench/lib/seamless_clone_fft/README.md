# seamless_clone_fft

OpenCV 4.9 `NORMAL_CLONE` reimplementation. Verified `maxDiff=0`, ~100ms on device (vs ~185ms baseline).

**Stable tag:** `seamless-fft-100ms-pass` (points at this DST merge + 3-channel path)

## Usage

```cpp
#include "seamless_clone_fft.h"

seamless_clone_fft::Context ctx;
ctx.seamlessClone(src, dst, mask, center, output);
```

## What makes ~100ms

1. **`cv::merge` + `cv::dft` + `cv::split`** DST path (faster than manual complex fill on device)
2. **`cv::parallel_for_` over 3 RGB channels** with `cv::setNumThreads(n/3)` per channel
3. Context buffer reuse + NEON prep/eigen/clamp

Do **not** replace `cv::dft` with third-party FFT (breaks maxDiff=0).

## Files

- `seamless_clone_fft.h` / `seamless_clone_fft.cpp`

Link OpenCV **core + imgproc** only.
