// 1D grid-stride enhance with half16 + inv by value (host: inv=1/fmin(1,max)).
// API: (src, width, height, float inv)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void enhanceBrightness(__global half* src, unsigned int width, unsigned int height,
                                float inv) {
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint n = width * height;
    const half hinv = (half)inv;

    const uint n16 = n >> 4;
    for (uint i = gid; i < n16; i += gsize) {
        half16 h = vload16(i, src);
        vstore16(h * hinv, i, src);
    }
    for (uint i = (n16 << 4) + gid; i < n; i += gsize) {
        src[i] = src[i] * hinv;
    }
}
