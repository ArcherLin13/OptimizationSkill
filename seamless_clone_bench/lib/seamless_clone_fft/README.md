# seamless_clone_fft

OpenCV 4.9 `NORMAL_CLONE` reimplementation (DST/DFT). Verified `maxDiff=0` vs `cv::seamlessClone`.

## Production usage

```cpp
#include "seamless_clone_fft.h"

seamless_clone_fft::Context ctx;

void onFrame(...) {
    ctx.seamlessClone(src, dst, mask, center, output);
}
```

## Optimizations

- Reused ROI / gradient / DST buffers (`Context`)
- **3 RGB channels** solved in parallel (private DST scratch each)
- `cv::setNumThreads(n/3)` per channel for `cv::dft`
- DST path avoids merge/split (direct complex layout)
- ARM NEON for padding, eigen divide, uchar clamp

Row FFT uses `cv::dft` (must match OpenCV exactly). Third-party FFTs (e.g. pocketfft) break `maxDiff=0` on device.

## Files

- `seamless_clone_fft.h` / `seamless_clone_fft.cpp`
- `sc_fft_rows.h` / `sc_fft_rows.cpp` — thin `cv::dft` DFT_ROWS wrapper

Link OpenCV **core + imgproc** only.
