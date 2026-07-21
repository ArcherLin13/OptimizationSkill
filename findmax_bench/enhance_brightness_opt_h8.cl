// 1D grid-stride enhance with half8 + inv by value (host: inv=1/fmin(1,max)).
// API: (src, width, height, float inv)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void enhanceBrightness(__global half* src, unsigned int width, unsigned int height,
                                float inv) {
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint n = width * height;
    const half hinv = (half)inv;

    const uint n8 = n >> 3;
    for (uint i = gid; i < n8; i += gsize) {
        half8 h = vload8(i, src);
        vstore8(h * hinv, i, src);
    }
    for (uint i = (n8 << 3) + gid; i < n; i += gsize) {
        src[i] = src[i] * hinv;
    }
}
