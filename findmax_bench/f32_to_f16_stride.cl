// Opt f32->f16: fixed gws = LSIZE*nwg + grid-stride + float4/half4.
// Host: gws = lws * nwg (NOT n). Each WI: for (i=gid; i<n4; i+=gsize).
// API: (src, dst, n)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef LSIZE
#define LSIZE 256
#endif

__kernel void f32_to_f16(__global const float* src, __global half* dst, uint n) {
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);

    const uint n4 = n >> 2;
    for (uint i = gid; i < n4; i += gsize) {
        const float4 f = vload4(i, src);
        const half4 h = convert_half4(f);
        vstore4(h, i, dst);
    }
    for (uint i = (n4 << 2) + gid; i < n; i += gsize) {
        dst[i] = convert_half(src[i]);
    }
}
