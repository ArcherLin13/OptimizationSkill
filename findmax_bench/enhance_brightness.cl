// Baseline enhanceBrightness (separate kernel after findMaxValue).
// divisor = fmin(1.0f, max_value); src[i] /= divisor (in-place).
// 2D 1-px/WI to pair with orig findMaxValue launch style.
//
// API: (__global half* src, unsigned int width, unsigned int height, __global half* max_value)
// max_value: half in low 16 bits of a 4-byte buffer (same as findmax atomic layout).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void enhanceBrightness(__global half* src, unsigned int width, unsigned int height,
                                __global half* max_value) {
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    if ((unsigned int)gid_x >= width || (unsigned int)gid_y >= height) {
        return;
    }

    const uint max_bits = *(__global uint*)max_value;
    const float mv = (float)as_half((ushort)(max_bits & 0xffffu));
    const float div = fmin(1.0f, mv);
    const float inv = (div > 1e-7f) ? (1.0f / div) : 1.0f;

    const uint idx = (unsigned int)gid_y * width + (unsigned int)gid_x;
    src[idx] = (half)((float)src[idx] * inv);
}
