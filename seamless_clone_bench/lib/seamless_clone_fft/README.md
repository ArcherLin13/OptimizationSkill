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
- ARM NEON for DST padding, eigen divide, and uchar clamp rows
- `cv::parallel_for_` on those CPU loops (does not parallelize `cv::dft`)

Dominant cost is still `cv::dft` inside DST (~90% of time on 718×115 ROI). NEON shaves CPU-loop overhead only.

## Files

- `seamless_clone_fft.h`
- `seamless_clone_fft.cpp`

Link OpenCV **core + imgproc** only.
