// 2D enhanceBrightness with half4 (vload4/vstore4) along X.
// Each WI owns 4 consecutive pixels: gid_x indexes groups of 4, gid_y is row.
// Launch: gws_x ~= ceil(width/4), gws_y ~= height (padded to local size).
// API: (src, width, height, float max_value)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void enhanceBrightness(__global half* src, unsigned int width, unsigned int height,
                                float max_value) {
    const uint x4 = get_global_id(0);
    const uint y = get_global_id(1);
    if (y >= height) {
        return;
    }

    const uint x0 = x4 << 2;
    if (x0 >= width) {
        return;
    }

    const float div = fmin(1.0f, max_value);
    const float inv = (div > 1e-7f) ? (1.0f / div) : 1.0f;
    const uint base = y * width + x0;

    // half4 needs 8-byte align => base % 4 == 0 (true when width%4==0 or y*width%4==0).
    if ((x0 + 3u < width) && ((base & 3u) == 0u)) {
        half4 h = vload4(0, src + base);
        float4 f = convert_float4(h) * inv;
        vstore4(convert_half4(f), 0, src + base);
    } else {
        for (uint x = x0; x < width && x < x0 + 4u; ++x) {
            const uint idx = y * width + x;
            src[idx] = (half)((float)src[idx] * inv);
        }
    }
}
