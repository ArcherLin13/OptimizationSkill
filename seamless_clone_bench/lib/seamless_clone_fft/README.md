# seamless_clone_fft (Phase A)

OpenCV 4.9 `NORMAL_CLONE` reimplementation using the same DST/DFT Poisson solver.

**Validation:** `poisson_fft` in benchmark must show `maxDiff=0` vs `cv::seamlessClone`.

Only after that is confirmed on device should we attempt FFT plan caching or faster libraries (Phase B).

## Files

- `seamless_clone_fft.h`
- `seamless_clone_fft.cpp`

Depends on OpenCV **core + imgproc** (uses `cv::dft`, `cv::Laplacian`).

## Usage

```cpp
#include "seamless_clone_fft.h"
seamless_clone_fft::seamlessClone(src, dst, mask, center, output);
```
