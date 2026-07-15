#pragma once

#include <cstddef>

// Planar float R|G|B -> interleaved BGR float (OpenCV CV_32FC3), NEON when available.
void neon_planar_rgb_f32_to_cv32fc3(const float* r, const float* g, const float* b, float* dst,
                                    int width, int height, int src_stride, int dst_stride);

// Planar uchar R|G|B -> interleaved BGR float (/255), NEON when available.
void neon_planar_rgb_u8_to_cv32fc3(const unsigned char* r, const unsigned char* g,
                                   const unsigned char* b, float* dst, int width, int height,
                                   int src_stride, int dst_stride);

// True if this build used NEON intrinsics (AArch64).
bool neon_planar_available();

// Multi-thread wrappers (row-parallel). threads<=0 uses hardware_concurrency.
void neon_planar_rgb_f32_to_cv32fc3_mt(const float* r, const float* g, const float* b, float* dst,
                                       int width, int height, int src_stride, int dst_stride,
                                       int threads);

void neon_planar_rgb_u8_to_cv32fc3_mt(const unsigned char* r, const unsigned char* g,
                                      const unsigned char* b, float* dst, int width, int height,
                                      int src_stride, int dst_stride, int threads);
