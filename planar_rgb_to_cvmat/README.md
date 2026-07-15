# Planar RGB → `cv::Mat` (`CV_32FC3`) OpenCL kernel

Converts **planar** RGB (full R plane, then G, then B) into OpenCV **interleaved** `CV_32FC3`.

## Layout

| | Memory |
|---|--------|
| **Input** | `R[0..N)`, `G[0..N)`, `B[0..N)` contiguous planes (`N = height * stride`) |
| **Output** | OpenCV `CV_32FC3`: per pixel `[B, G, R]` float (default) |

## Kernels

| Kernel | Use when |
|--------|----------|
| `planar_rgb_to_cv32fc3` | Separate `r`, `g`, `b` buffers |
| `planar_rgb_packed_to_cv32fc3` | One buffer: `[R\|G\|B]` |

## Host sketch

```cpp
cv::Mat dst(height, width, CV_32FC3);  // preferably continuous
// cl buffers: planar float or uchar; dst is float*, size width*height*3

size_t global[2] = { (size_t)width, (size_t)height };
clSetKernelArg(..., 4, sizeof(int), &width);
clSetKernelArg(..., 5, sizeof(int), &height);
int src_stride = width;           // tightly packed planes
int dst_stride = (int)(dst.step / sizeof(float));  // = width*3 if continuous
clEnqueueNDRangeKernel(..., 2, nullptr, global, nullptr, ...);
// map or clEnqueueReadBuffer into dst.data
```

## Compile flags

| Flag | Effect |
|------|--------|
| *(none)* | `float` planes → float BGR |
| `-DINPUT_UCHAR` | `uchar` planes, scale `/255` |
| `-DOUT_RGB` | write RGB instead of OpenCV BGR |

## Note

OpenCV color mats use **BGR** channel order. If your pipeline expects RGB floats, build with `-DOUT_RGB`.
