// Baseline enhanceBrightness (separate kernel after findMaxValue).
// divisor = fmin(1.0f, max_value); src[i] *= 1/divisor (in-place).
// 2D 1-px/WI to pair with orig findMaxValue launch style.
//
// API: (__global half* src, unsigned int width, unsigned int height, float max_value)
// max_value: host-passed float (read findmax result on host, then pass by value).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void enhanceBrightness(__global half* src, unsigned int width, unsigned int height,
                                float max_value) {
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    if ((unsigned int)gid_x >= width || (unsigned int)gid_y >= height) {
        return;
    }

    const float div = fmin(1.0f, max_value);
    const float inv = (div > 1e-7f) ? (1.0f / div) : 1.0f;

    const uint idx = (unsigned int)gid_y * width + (unsigned int)gid_x;
    src[idx] = (half)((float)src[idx] * inv);
}
