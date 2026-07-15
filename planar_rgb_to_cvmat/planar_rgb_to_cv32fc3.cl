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
//
// NDRange: global = (width, height)

#ifndef CHANNEL_ORDER_BGR
#define CHANNEL_ORDER_BGR 1
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

// 2D: get_global_id(0)=x, get_global_id(1)=y
__kernel void planar_rgb_to_cv32fc3(
    __global const src_t* restrict r,
    __global const src_t* restrict g,
    __global const src_t* restrict b,
    __global float* restrict dst,  // CV_32FC3 continuous: width*height*3 floats
    const int width,
    const int height,
    const int src_stride,  // elements per row (>= width); pass width if tightly packed
    const int dst_stride   // floats per row (>= width*3); pass width*3 if continuous Mat
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    const int si = y * src_stride + x;
    const float rv = load_scaled(r, si);
    const float gv = load_scaled(g, si);
    const float bv = load_scaled(b, si);

    __global float* out = dst + y * dst_stride + x * 3;
#if defined(OUT_RGB)
    out[0] = rv;
    out[1] = gv;
    out[2] = bv;
#else
    // OpenCV default color order for CV_8UC3 / CV_32FC3
    out[0] = bv;
    out[1] = gv;
    out[2] = rv;
#endif
}

// Packed planar buffer: [R plane | G plane | B plane], each height*src_stride
__kernel void planar_rgb_packed_to_cv32fc3(
    __global const src_t* restrict planar,  // size >= 3 * height * src_stride
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
    const float rv = load_scaled(planar + 0 * plane, si);
    const float gv = load_scaled(planar + 1 * plane, si);
    const float bv = load_scaled(planar + 2 * plane, si);

    __global float* out = dst + y * dst_stride + x * 3;
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
