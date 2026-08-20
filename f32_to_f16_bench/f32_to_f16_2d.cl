// Baseline f32->f16: 2D NDRange, gws ~= (width, height) — classic image launch.
// API: (src, dst, width, height)  flat index = y*width + x

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void f32_to_f16(__global const float* src, __global half* dst, uint width,
                         uint height) {
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }
    const uint idx = y * width + x;
    dst[idx] = convert_half(src[idx]);
}
