# seamless_clone_fft

OpenCV 4.9 `NORMAL_CLONE` reimplementation (DST/DFT). Verified `maxDiff=0` vs `cv::seamlessClone`.

## Production usage (fast path)

```cpp
#include "seamless_clone_fft.h"

// Reuse Context across frames (buffer pool + parallel RGB channels).
seamless_clone_fft::Context ctx;

void onFrame(...) {
    ctx.seamlessClone(src, dst, mask, center, output);

    // Optional: skip when inputs unchanged (same output, ~0ms).
  // seamless_clone_fft::SkipState skip;
  // if (seamless_clone_fft::seamlessCloneSkipUnchanged(skip, src, dst, mask, center, output, 1, &ctx))
  //     return;
}
```

## Phase B optimizations

- Reused ROI / gradient / DST buffers (`Context`)
- `cv::parallel_for_` over 3 color channels (identical math)
- Optional `seamlessCloneSkipUnchanged` for camera pipelines

Still uses `cv::dft` — expect modest speedup vs OpenCV photo; largest gain when frames repeat.

## Files

- `seamless_clone_fft.h`
- `seamless_clone_fft.cpp`

Link OpenCV **core + imgproc** only.
