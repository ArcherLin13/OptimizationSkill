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
- **3 RGB channels solved in parallel** (separate DST scratch per channel)
- `cv::setNumThreads(n/3)` per channel so `cv::dft` does not oversubscribe
- ARM NEON for DST padding, eigen divide, uchar clamp

Dominant cost is still `cv::dft`. Expect ~1.5–2.5× vs OpenCV `seamlessClone` on multi-core ARM when maxDiff=0.

## Files

- `seamless_clone_fft.h`
- `seamless_clone_fft.cpp`

Link OpenCV **core + imgproc** only.
