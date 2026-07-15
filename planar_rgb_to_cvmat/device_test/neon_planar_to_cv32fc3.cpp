#include "neon_planar_to_cv32fc3.h"

#include <algorithm>
#include <thread>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>
#define PLANAR_HAS_NEON 1
#else
#define PLANAR_HAS_NEON 0
#endif

bool neon_planar_available() {
#if PLANAR_HAS_NEON
    return true;
#else
    return false;
#endif
}

#if PLANAR_HAS_NEON

namespace {

inline void store_bgr_tail(const float* r, const float* g, const float* b, float* dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst[0] = b[i];
        dst[1] = g[i];
        dst[2] = r[i];
        dst += 3;
    }
}

inline void convert_row_f32(const float* r, const float* g, const float* b, float* dst, int width) {
    int x = 0;
    for (; x + 4 <= width; x += 4) {
        float32x4x3_t v;
        v.val[0] = vld1q_f32(b + x);  // B
        v.val[1] = vld1q_f32(g + x);  // G
        v.val[2] = vld1q_f32(r + x);  // R
        vst3q_f32(dst + x * 3, v);
    }
    store_bgr_tail(r + x, g + x, b + x, dst + x * 3, width - x);
}

inline void convert_row_u8(const unsigned char* r, const unsigned char* g, const unsigned char* b,
                           float* dst, int width) {
    const float32x4_t scale = vdupq_n_f32(1.f / 255.f);
    int x = 0;
    // 8 pixels: load u8 -> u16 -> two f32x4, then two vst3q
    for (; x + 8 <= width; x += 8) {
        const uint8x8_t ru8 = vld1_u8(r + x);
        const uint8x8_t gu8 = vld1_u8(g + x);
        const uint8x8_t bu8 = vld1_u8(b + x);

        const uint16x8_t ru16 = vmovl_u8(ru8);
        const uint16x8_t gu16 = vmovl_u8(gu8);
        const uint16x8_t bu16 = vmovl_u8(bu8);

        float32x4x3_t lo;
        lo.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(bu16))), scale);
        lo.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(gu16))), scale);
        lo.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(ru16))), scale);
        vst3q_f32(dst + x * 3, lo);

        float32x4x3_t hi;
        hi.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(bu16))), scale);
        hi.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(gu16))), scale);
        hi.val[2] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(ru16))), scale);
        vst3q_f32(dst + (x + 4) * 3, hi);
    }
    for (; x < width; ++x) {
        dst[x * 3 + 0] = b[x] * (1.f / 255.f);
        dst[x * 3 + 1] = g[x] * (1.f / 255.f);
        dst[x * 3 + 2] = r[x] * (1.f / 255.f);
    }
}

}  // namespace

void neon_planar_rgb_f32_to_cv32fc3(const float* r, const float* g, const float* b, float* dst,
                                    int width, int height, int src_stride, int dst_stride) {
    for (int y = 0; y < height; ++y) {
        convert_row_f32(r + y * src_stride, g + y * src_stride, b + y * src_stride,
                        dst + y * dst_stride, width);
    }
}

void neon_planar_rgb_u8_to_cv32fc3(const unsigned char* r, const unsigned char* g,
                                   const unsigned char* b, float* dst, int width, int height,
                                   int src_stride, int dst_stride) {
    for (int y = 0; y < height; ++y) {
        convert_row_u8(r + y * src_stride, g + y * src_stride, b + y * src_stride,
                       dst + y * dst_stride, width);
    }
}

#else

void neon_planar_rgb_f32_to_cv32fc3(const float* r, const float* g, const float* b, float* dst,
                                    int width, int height, int src_stride, int dst_stride) {
    for (int y = 0; y < height; ++y) {
        const float* rr = r + y * src_stride;
        const float* gg = g + y * src_stride;
        const float* bb = b + y * src_stride;
        float* out = dst + y * dst_stride;
        for (int x = 0; x < width; ++x) {
            out[0] = bb[x];
            out[1] = gg[x];
            out[2] = rr[x];
            out += 3;
        }
    }
}

void neon_planar_rgb_u8_to_cv32fc3(const unsigned char* r, const unsigned char* g,
                                   const unsigned char* b, float* dst, int width, int height,
                                   int src_stride, int dst_stride) {
    constexpr float k = 1.f / 255.f;
    for (int y = 0; y < height; ++y) {
        const unsigned char* rr = r + y * src_stride;
        const unsigned char* gg = g + y * src_stride;
        const unsigned char* bb = b + y * src_stride;
        float* out = dst + y * dst_stride;
        for (int x = 0; x < width; ++x) {
            out[0] = bb[x] * k;
            out[1] = gg[x] * k;
            out[2] = rr[x] * k;
            out += 3;
        }
    }
}

#endif

void neon_planar_rgb_f32_to_cv32fc3_mt(const float* r, const float* g, const float* b, float* dst,
                                       int width, int height, int src_stride, int dst_stride,
                                       int threads) {
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    threads = std::max(1, std::min(threads, height));
    if (threads == 1) {
        neon_planar_rgb_f32_to_cv32fc3(r, g, b, dst, width, height, src_stride, dst_stride);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const int y0 = height * t / threads;
        const int y1 = height * (t + 1) / threads;
        const int rows = y1 - y0;
        workers.emplace_back([=] {
            neon_planar_rgb_f32_to_cv32fc3(r + y0 * src_stride, g + y0 * src_stride,
                                           b + y0 * src_stride, dst + y0 * dst_stride, width, rows,
                                           src_stride, dst_stride);
        });
    }
    for (auto& th : workers) {
        th.join();
    }
}

void neon_planar_rgb_u8_to_cv32fc3_mt(const unsigned char* r, const unsigned char* g,
                                      const unsigned char* b, float* dst, int width, int height,
                                      int src_stride, int dst_stride, int threads) {
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    threads = std::max(1, std::min(threads, height));
    if (threads == 1) {
        neon_planar_rgb_u8_to_cv32fc3(r, g, b, dst, width, height, src_stride, dst_stride);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const int y0 = height * t / threads;
        const int y1 = height * (t + 1) / threads;
        const int rows = y1 - y0;
        workers.emplace_back([=] {
            neon_planar_rgb_u8_to_cv32fc3(r + y0 * src_stride, g + y0 * src_stride,
                                          b + y0 * src_stride, dst + y0 * dst_stride, width, rows,
                                          src_stride, dst_stride);
        });
    }
    for (auto& th : workers) {
        th.join();
    }
}
