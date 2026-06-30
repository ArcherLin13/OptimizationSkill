#pragma once

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

inline float32x2_t sc_mul_complex2(float32x2_t a, float32x2_t w) {
    const float ar = vget_lane_f32(a, 0);
    const float ai = vget_lane_f32(a, 1);
    const float wr = vget_lane_f32(w, 0);
    const float wi = vget_lane_f32(w, 1);
    float32x2_t r = vdup_n_f32(0.f);
    r = vset_lane_f32(ar * wr - ai * wi, r, 0);
    r = vset_lane_f32(ar * wi + ai * wr, r, 1);
    return r;
}

inline float32x4_t sc_mul_complex4(float32x4_t a, float32x4_t w) {
    return vcombine_f32(sc_mul_complex2(vget_low_f32(a), vget_low_f32(w)),
                        sc_mul_complex2(vget_high_f32(a), vget_high_f32(w)));
}

template <>
struct DFT_VecR4<float> {
    int operator()(Complex<float>* dst, int N, int n0, int& _dw0, const Complex<float>* wave) const {
        int n = 1;
        int i = 0;
        int j = 0;
        int nx = 0;
        int dw = 0;
        int dw0 = _dw0;
        const float32x4_t z = vdupq_n_f32(0.f);
        float32x4_t x02 = z;
        float32x4_t x13 = z;
        float32x4_t y01 = z;
        float32x4_t y23 = z;
        float32x4_t t0 = z;
        float32x4_t t1 = z;
        float32x4_t w23 = z;

        for (; n * 4 <= N;) {
            nx = n;
            n *= 4;
            dw0 /= 4;

            for (i = 0; i < n0; i += n) {
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

                for (j = 1, dw = dw0; j < nx; j++, dw += dw0) {
                    v0 = dst + i + j;
                    v1 = v0 + nx * 2;

                    x13 = sc_vld2cx(&v0[nx], &v1[nx]);
                    w23 = sc_vld2cx(&wave[dw * 2], &wave[dw * 3]);
                    x13 = sc_mul_complex4(x13, w23);

                    const float32x2_t x2 = vld1_f32(reinterpret_cast<float*>(&v1[0]));
                    const float32x2_t w1 = vld1_f32(reinterpret_cast<const float*>(&wave[dw]));
                    const float32x2_t x0 = vld1_f32(reinterpret_cast<float*>(&v0[0]));
                    x02 = vcombine_f32(x0, sc_mul_complex2(x2, w1));

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
