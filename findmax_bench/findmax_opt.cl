// OCR-softmax-style findMaxValue (same API as orig):
//   (__global half* src, unsigned int width, unsigned int height, __global half* max_value)
//
// Launch (host): gws = LSIZE * nwg, lws == LSIZE  (must match -DLSIZE).
//
// Image bounds (flat layout, n = width*height):
//   - half4: vload4(i) touches [4*i, 4*i+3]; loop only while i < (n>>2)
//            => last index = 4*(n>>2)-1 = (n&~3)-1 < n
 //   - scalar tail: for (i = (n&~3)+gid; i < n; i += gsize)
//   - n==0: no loads; atomic gets -inf lane (no-op vs typical init)
// Host must ensure buffer bytes >= n*sizeof(half) and n fits in uint.

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
    const uint lsize = get_local_size(0);
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);

    // Mis-launch (lws != LSIZE): all WIs take this path (same lsize) — no barrier deadlock.
    if (lsize != (uint)LSIZE) {
        return;
    }

    // Use ulong multiply; if product exceeds uint addressing, treat as empty (host should reject).
    const ulong n64 = (ulong)width * (ulong)height;
    const uint n = (n64 > 0xffffffffUL) ? 0u : (uint)n64;

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

    for (uint stride = (uint)LSIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            reduce_buf[lid] = fmax(reduce_buf[lid], reduce_buf[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_max_half(max_value, (half)reduce_buf[0]);
    }
}
