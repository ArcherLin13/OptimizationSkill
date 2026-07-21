// OCR-softmax-style findMaxValue (same API as orig):
//   (__global half* src, unsigned int width, unsigned int height, __global half* max_value)
//
// vs 1-px/WI baseline:
//   - grid-stride: each WI scans many pixels (like softmax_ocr_opt_2d lane loop)
//   - private lane_max, then local tree reduce
//   - only 1 atomic per workgroup (not per pixel-group of 1)
//   - half4 vector loads for coalesced bandwidth
//
// Launch (host): gws = lws * nwg, e.g. local=256, nwg=128..512 (NOT width*height).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef LSIZE
#define LSIZE 256
#endif

inline void atomic_max_half(__global half* addr, half val) {
    volatile __global uint* up = (volatile __global uint*)addr;
    uint old = *up;
    for (;;) {
        const half old_h = as_half((ushort)(old & 0xffffu));
        if ((float)val <= (float)old_h) {
            return;
        }
        const uint neu = (old & 0xffff0000u) | (uint)as_ushort(val);
        const uint prev = atom_cmpxchg(up, old, neu);
        if (prev == old) {
            return;
        }
        old = prev;
    }
}

__kernel void findMaxValue(__global half* src, unsigned int width, unsigned int height,
                           __global half* max_value) {
    __local float reduce_buf[LSIZE];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint n = width * height;

    float lane_max = -65504.0f;

    // Vectorized grid-stride over half4 groups (coalesced).
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
        atomic_max_half(max_value, (half)reduce_buf[0]);
    }
}
