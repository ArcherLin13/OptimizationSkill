// Opt enhanceBrightness: same launch style as findmax_opt (grid-stride + half4).
// divisor = fmin(1.0f, max_value); src *= 1/divisor (in-place).
// API: (src, width, height, max_value) — max_value uses 4B half atomic layout.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef LSIZE
#define LSIZE 256
#endif

__kernel void enhanceBrightness(__global half* src, unsigned int width, unsigned int height,
                                __global half* max_value) {
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint n = width * height;

    const uint max_bits = *(__global uint*)max_value;
    const float mv = (float)as_half((ushort)(max_bits & 0xffffu));
    const float div = fmin(1.0f, mv);
    const float inv = (div > 1e-7f) ? (1.0f / div) : 1.0f;

    const uint n4 = n >> 2;
    for (uint i = gid; i < n4; i += gsize) {
        half4 h = vload4(i, src);
        float4 f = convert_float4(h) * inv;
        vstore4(convert_half4(f), i, src);
    }
    for (uint i = (n4 << 2) + gid; i < n; i += gsize) {
        src[i] = (half)((float)src[i] * inv);
    }
}
