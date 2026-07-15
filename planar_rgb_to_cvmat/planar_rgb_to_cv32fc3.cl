// Planar RGB (R-plane | G-plane | B-plane) -> OpenCV CV_32FC3 (interleaved BGR float).
//
// Input layout (height * width floats or uchars each plane):
//   r[0..N), g[0..N), b[0..N)   N = height * width  (contiguous planes)
// Output layout (OpenCV Mat CV_32FC3, continuous):
//   dst[i*3+0]=B, dst[i*3+1]=G, dst[i*3+2]=R
//
// Compile options (optional):
//   -DINPUT_UCHAR     input planes are uchar; scale by 1/255 to float
//   -DOUT_RGB         write RGB instead of OpenCV default BGR
//   -DPIXELS_PER_WI=N  pixels per work-item for *_ppx kernels (default 8)
//
// NDRange:
//   1px kernels: global = (width, height)
//   ppx kernels: global = (ceil(width / PIXELS_PER_WI), height)

#ifndef PIXELS_PER_WI
#define PIXELS_PER_WI 8
#endif

#ifdef INPUT_UCHAR
typedef uchar src_t;
__constant float kScale = 1.f / 255.f;
#else
typedef float src_t;
__constant float kScale = 1.f;
#endif

inline float load_scaled(__global const src_t* p, int i) {
#ifdef INPUT_UCHAR
    return convert_float(p[i]) * kScale;
#else
    return p[i] * kScale;
#endif
}

inline void store_pixel(__global float* out, float rv, float gv, float bv) {
#if defined(OUT_RGB)
    out[0] = rv;
    out[1] = gv;
    out[2] = bv;
#else
    out[0] = bv;
    out[1] = gv;
    out[2] = rv;
#endif
}

// ---- baseline: 1 pixel / work-item -----------------------------------------

__kernel void planar_rgb_to_cv32fc3(
    __global const src_t* restrict r,
    __global const src_t* restrict g,
    __global const src_t* restrict b,
    __global float* restrict dst,
    const int width,
    const int height,
    const int src_stride,
    const int dst_stride
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int si = y * src_stride + x;
    store_pixel(dst + y * dst_stride + x * 3, load_scaled(r, si), load_scaled(g, si),
                load_scaled(b, si));
}

__kernel void planar_rgb_packed_to_cv32fc3(
    __global const src_t* restrict planar,
    __global float* restrict dst,
    const int width,
    const int height,
    const int src_stride,
    const int dst_stride
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const int plane = height * src_stride;
    const int si = y * src_stride + x;
    store_pixel(dst + y * dst_stride + x * 3, load_scaled(planar + 0 * plane, si),
                load_scaled(planar + 1 * plane, si), load_scaled(planar + 2 * plane, si));
}

// ---- optimized: PIXELS_PER_WI pixels / work-item + vector loads ------------

inline void store4(__global float* out, float4 rv, float4 gv, float4 bv) {
    store_pixel(out + 0, rv.s0, gv.s0, bv.s0);
    store_pixel(out + 3, rv.s1, gv.s1, bv.s1);
    store_pixel(out + 6, rv.s2, gv.s2, bv.s2);
    store_pixel(out + 9, rv.s3, gv.s3, bv.s3);
}

inline float4 load4_scaled(__global const src_t* p, int i) {
#ifdef INPUT_UCHAR
    return convert_float4(vload4(0, p + i)) * kScale;
#else
    return vload4(0, p + i) * kScale;
#endif
}

// Process one row chunk starting at (x0,y). Handles width remainder.
inline void convert_ppx_chunk(__global const src_t* restrict r, __global const src_t* restrict g,
                              __global const src_t* restrict b, __global float* restrict dst,
                              const int width, const int src_stride, const int dst_stride,
                              const int x0, const int y) {
    const int row_src = y * src_stride;
    const int row_dst = y * dst_stride;
    int x = x0;
    const int x_end = min(x0 + PIXELS_PER_WI, width);

    // Vector path: blocks of 4
    while (x + 4 <= x_end) {
        const int si = row_src + x;
        const float4 rv = load4_scaled(r, si);
        const float4 gv = load4_scaled(g, si);
        const float4 bv = load4_scaled(b, si);
        store4(dst + row_dst + x * 3, rv, gv, bv);
        x += 4;
    }
    // Tail
    while (x < x_end) {
        const int si = row_src + x;
        store_pixel(dst + row_dst + x * 3, load_scaled(r, si), load_scaled(g, si),
                    load_scaled(b, si));
        ++x;
    }
}

__kernel void planar_rgb_to_cv32fc3_ppx(
    __global const src_t* restrict r,
    __global const src_t* restrict g,
    __global const src_t* restrict b,
    __global float* restrict dst,
    const int width,
    const int height,
    const int src_stride,
    const int dst_stride
) {
    const int x0 = (int)get_global_id(0) * PIXELS_PER_WI;
    const int y = (int)get_global_id(1);
    if (y >= height || x0 >= width) {
        return;
    }
    convert_ppx_chunk(r, g, b, dst, width, src_stride, dst_stride, x0, y);
}

__kernel void planar_rgb_packed_to_cv32fc3_ppx(
    __global const src_t* restrict planar,
    __global float* restrict dst,
    const int width,
    const int height,
    const int src_stride,
    const int dst_stride
) {
    const int x0 = (int)get_global_id(0) * PIXELS_PER_WI;
    const int y = (int)get_global_id(1);
    if (y >= height || x0 >= width) {
        return;
    }
    const int plane = height * src_stride;
    convert_ppx_chunk(planar + 0 * plane, planar + 1 * plane, planar + 2 * plane, dst, width,
                      src_stride, dst_stride, x0, y);
}
