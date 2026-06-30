# seamless_clone_fft

OpenCV 4.9 `NORMAL_CLONE` reimplementation (DST/DFT). Verified `maxDiff=0` vs `cv::seamlessClone`.

**Stable release:** git tag `seamless-fft-100ms-pass` (~100ms on device, maxDiff=0).

## Production usage

```cpp
#include "seamless_clone_fft.h"

seamless_clone_fft::Context ctx;
ctx.seamlessClone(src, dst, mask, center, output);
```

## Optimizations (verified)

- Reused ROI / gradient / DST buffers (`Context`)
- **3 RGB channels** solved in parallel (private DST scratch each)
- `cv::setNumThreads(n/3)` per channel
- DST path avoids merge/split; ARM NEON for prep / eigen / clamp
- Row FFT uses **`cv::dft`** (required for maxDiff=0; ported FFT/NEON failed on device)

## Files

- `seamless_clone_fft.h` / `seamless_clone_fft.cpp`
- `sc_fft_rows.h` / `sc_fft_rows.cpp`

Link OpenCV **core + imgproc** only.
