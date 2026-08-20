// Baseline f32->f16: 1D NDRange, gws ~= n (1 element / WI).
// API: (src, dst, n)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void f32_to_f16(__global const float* src, __global half* dst, uint n) {
    const uint i = get_global_id(0);
    if (i < n) {
        dst[i] = convert_half(src[i]);
    }
}
