# seamless_clone_jacobi

CPU-only drop-in replacement for `cv::seamlessClone(..., NORMAL_CLONE)`.

## Copy to your project

Copy this **folder** as-is (3 files):

```
seamless_clone_jacobi/
  seamless_clone_jacobi.h
  seamless_roi.h
  seamless_clone_jacobi.cpp
```

Add `seamless_clone_jacobi.cpp` to your build. Link **OpenCV core + imgproc** (no `opencv_photo`, no OpenCL, no `dl`).

## Usage

```cpp
#include "seamless_clone_jacobi.h"

// cv::seamlessClone(src, dst, mask, center, output, cv::NORMAL_CLONE);
seamless_clone_jacobi::seamlessClone(src, dst, mask, center, output);

// optional: iterations (default 400)
seamless_clone_jacobi::seamlessClone(src, dst, mask, center, output, 1, 400);
```

## Notes

- Same inputs as OpenCV: `src`/`dst` `CV_8UC3`, `mask` `CV_8U`, `center` in dst coordinates.
- Approximate result (~3x faster than OpenCV FFT on 729x126 rect mask). Default 400 iterations is the recommended tradeoff.
