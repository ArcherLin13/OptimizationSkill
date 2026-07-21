// True 1-kernel findMax + enhance (NO cross-WG spin — that caused CL_-14 on device).
// Launch with a SINGLE workgroup: gws == lws == LSIZE (nwg=1).
// Phase1: grid-stride half4 max + local reduce
// Phase2: same WI enhance in-place (local barrier only)
//
// API: (src, width, height, max_value)  — same 4 args as findmax (no sync buffer)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef LSIZE
#define LSIZE 256
#endif

__kernel void findMaxAndEnhance(__global half* src, unsigned int width, unsigned int height,
                                __global half* max_value) {
    __local float reduce_buf[LSIZE];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint n = width * height;

    // ----- Phase 1: max -----
    float lane_max = -65504.0f;
    const uint n4 = n >> 2;
    for (uint i = gid; i < n4; i += gsize) {
        const half4 h = vload4(i, src);
        const float4 f = convert_float4(h);
        lane_max = fmax(lane_max, fmax(fmax(f.s0, f.s1), fmax(f.s2, f.s3)));
    }
    for (uint i = (n4 << 2) + gid; i < n; i += gsize) {
        lane_max = fmax(lane_max, (float)src[i]);
    }

    reduce_buf[lid] = lane_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = LSIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            reduce_buf[lid] = fmax(reduce_buf[lid], reduce_buf[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        // Publish max (4B layout, low 16 = half)
        volatile __global uint* up = (volatile __global uint*)max_value;
        *up = (uint)as_ushort((half)reduce_buf[0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ----- Phase 2: enhance -----
    const float mv = reduce_buf[0];
    const float div = fmin(1.0f, mv);
    const float inv = (div > 1e-7f) ? (1.0f / div) : 1.0f;

    for (uint i = gid; i < n4; i += gsize) {
        half4 h = vload4(i, src);
        float4 f = convert_float4(h) * inv;
        vstore4(convert_half4(f), i, src);
    }
    for (uint i = (n4 << 2) + gid; i < n; i += gsize) {
        src[i] = (half)((float)src[i] * inv);
    }
}
