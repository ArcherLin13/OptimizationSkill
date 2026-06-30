# seamless_clone_fft

OpenCV 4.9 `NORMAL_CLONE` reimplementation (DST/DFT). Target: `maxDiff=0` vs `cv::seamlessClone`.

## Production usage

```cpp
#include "seamless_clone_fft.h"

seamless_clone_fft::Context ctx;

void onFrame(...) {
    ctx.seamlessClone(src, dst, mask, center, output);
}
```

## Optimizations

- **pocketfft** replaces `cv::dft` in DST (cached plans, better factors on 1434/228 lengths)
- **3 RGB channels** solved in parallel (private DST scratch each)
- `cv::setNumThreads(n/3)` per channel
- ARM NEON for DST padding, eigen divide, uchar clamp
- Context buffer reuse

## Files

- `seamless_clone_fft.h` / `seamless_clone_fft.cpp`
- `sc_fft_rows.h` / `sc_fft_rows.cpp` — row FFT wrapper
- `pocketfft_hdronly.h` — vendored pocketfft (BSD-3-Clause)

Link OpenCV **core + imgproc** only.
