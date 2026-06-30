#pragma once

// ARM NEON port of OpenCV dxt.cpp DFT_VecR4<float> (SSE3), lane layout must match exactly.

#if defined(__aarch64__) && defined(SC_OCV_DFT_NEON)

inline float32x4_t sc_vld2cx(const Complex<float>* a, const Complex<float>* b) {
    return vcombine_f32(vld1_f32(reinterpret_cast<const float*>(a)),
                        vld1_f32(reinterpret_cast<const float*>(b)));
}

inline void sc_vstl2cx(Complex<float>* dst, float32x4_t v) {
    vst1_f32(reinterpret_cast<float*>(dst), vget_low_f32(v));
}

inline void sc_vsth2cx(Complex<float>* dst, float32x4_t v) {
    vst1_f32(reinterpret_cast<float*>(dst), vget_high_f32(v));
}

inline float32x4_t sc_shuffle2332(float32x4_t y01, float32x4_t y23) {
    const float32x2_t lo = vget_high_f32(y01);
    const float32x2_t hi = vrev64_f32(vget_high_f32(y23));
    return vcombine_f32(lo, hi);
}

inline float32x4_t sc_movelh(float32x4_t a, float32x4_t b) {
    return vcombine_f32(vget_low_f32(a), vget_low_f32(b));
}

inline float32x4_t sc_vxor_sign_lane3(float32x4_t v) {
    const uint32x4_t mask = {0u, 0u, 0u, 0x80000000u};
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), mask));
}

inline float32x4_t sc_moveldup(float32x4_t v) {
    return vcombine_f32(vdup_lane_f32(vget_low_f32(v), 0), vdup_lane_f32(vget_high_f32(v), 0));
}

inline float32x4_t sc_movehdup(float32x4_t v) {
    return vcombine_f32(vdup_lane_f32(vget_low_f32(v), 1), vdup_lane_f32(vget_high_f32(v), 1));
}

inline float32x4_t sc_shuffle2301(float32x4_t v) {
    return (float32x4_t){vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3), vgetq_lane_f32(v, 0),
                         vgetq_lane_f32(v, 1)};
}

inline float32x4_t sc_addsub(float32x4_t a, float32x4_t b) {
    const uint32x4_t flip = {0u, 0x80000000u, 0u, 0x80000000u};
    return vaddq_f32(a, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(b), flip)));
}

inline float32x4_t sc_load_complex_as_ps(const Complex<float>* c) {
    return vld1q_f32(reinterpret_cast<const float*>(c));
}

inline float32x4_t sc_shuffle1100(float32x4_t v) {
    return (float32x4_t){vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1),
                         vgetq_lane_f32(v, 1)};
}

inline float32x4_t sc_shuffle1001(float32x4_t v) {
    return (float32x4_t){vgetq_lane_f32(v, 1), vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 0),
                         vgetq_lane_f32(v, 1)};
}

template <>
struct DFT_VecR4<float> {
    int operator()(Complex<float>* dst, int N, int n0, int& _dw0, const Complex<float>* wave) const {
        int n = 1;
        int dw0 = _dw0;
        const float32x4_t z = vdupq_n_f32(0.f);
        float32x4_t x02 = z;
        float32x4_t x13 = z;
        float32x4_t w01 = z;
        float32x4_t w23 = z;
        float32x4_t y01 = z;
        float32x4_t y23 = z;
        float32x4_t t0 = z;
        float32x4_t t1 = z;

        for (; n * 4 <= N;) {
            const int nx = n;
            n *= 4;
            dw0 /= 4;

            for (int i = 0; i < n0; i += n) {
                Complex<float>* v0 = dst + i;
                Complex<float>* v1 = v0 + nx * 2;

                x02 = sc_vld2cx(&v0[0], &v1[0]);
                x13 = sc_vld2cx(&v0[nx], &v1[nx]);

                y01 = vaddq_f32(x02, x13);
                y23 = vsubq_f32(x02, x13);
                t1 = sc_vxor_sign_lane3(sc_shuffle2332(y01, y23));
                t0 = sc_movelh(y01, y23);
                y01 = vaddq_f32(t0, t1);
                y23 = vsubq_f32(t0, t1);

                sc_vstl2cx(&v0[0], y01);
                sc_vsth2cx(&v0[nx], y01);
                sc_vstl2cx(&v1[0], y23);
                sc_vsth2cx(&v1[nx], y23);

                for (int j = 1, dw = dw0; j < nx; j++, dw += dw0) {
                    v0 = dst + i + j;
                    v1 = v0 + nx * 2;

                    x13 = sc_vld2cx(&v0[nx], &v1[nx]);
                    w23 = sc_vld2cx(&wave[dw * 2], &wave[dw * 3]);
                    t0 = vmulq_f32(sc_moveldup(x13), w23);
                    t1 = vmulq_f32(sc_movehdup(x13), sc_shuffle2301(w23));
                    x13 = sc_addsub(t0, t1);

                    x02 = sc_load_complex_as_ps(&v1[0]);
                    w01 = sc_load_complex_as_ps(&wave[dw]);
                    x02 = sc_shuffle1100(x02);
                    w01 = sc_shuffle1001(w01);
                    t0 = vmulq_f32(x02, w01);
                    x02 = sc_addsub(t0, sc_movelh(t0, t0));
                    x02 = vsetq_lane_f32(vgetq_lane_f32(sc_load_complex_as_ps(&v0[0]), 0), x02, 0);
                    x02 = vsetq_lane_f32(vgetq_lane_f32(sc_load_complex_as_ps(&v0[0]), 1), x02, 1);

                    y01 = vaddq_f32(x02, x13);
                    y23 = vsubq_f32(x02, x13);
                    t1 = sc_vxor_sign_lane3(sc_shuffle2332(y01, y23));
                    t0 = sc_movelh(y01, y23);
                    y01 = vaddq_f32(t0, t1);
                    y23 = vsubq_f32(t0, t1);

                    sc_vstl2cx(&v0[0], y01);
                    sc_vsth2cx(&v0[nx], y01);
                    sc_vstl2cx(&v1[0], y23);
                    sc_vsth2cx(&v1[nx], y23);
                }
            }
        }

        _dw0 = dw0;
        return n;
    }
};

#endif
