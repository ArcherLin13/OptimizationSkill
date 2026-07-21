// Fused findMax + enhanceBrightness in ONE kernel (based on opt_stride).
// Phase 1: grid-stride half4 max + local reduce + 1 atomic/WG
// Phase 2: grid sync (arrival counter) then in-place enhance with half4
//
// divisor = fmin(1.0f, max_value); src[i] *= 1/divisor
//
// Args:
//   src, width, height, max_value (4B half atomic layout),
//   sync: int2 — sync[0]=wg arrival (init 0), sync[1]=ready flag (init 0)
//
// Launch: gws = LSIZE * nwg (same as findmax_opt), NOT width*height.

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

__kernel void findMaxAndEnhance(__global half* src, unsigned int width, unsigned int height,
                                __global half* max_value, __global volatile int* sync) {
    __local float reduce_buf[LSIZE];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const uint nwg = get_num_groups(0);
    const uint n = width * height;

    // ----- Phase 1: find max (OCR-style stride + half4) -----
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
        atomic_max_half(max_value, (half)reduce_buf[0]);
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        // Arrive only after this WG's atomic_max is done.
        const int ticket = atom_inc((__global int*)&sync[0]);
        if (ticket == (int)nwg - 1) {
            atom_xchg((__global int*)&sync[1], 1);
        }
    }

    // Wait until all WGs published their max.
    while (atom_or((__global int*)&sync[1], 0) == 0) {
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    // ----- Phase 2: enhance in-place -----
    const uint max_bits = *(__global uint*)max_value;
    const float mv = (float)as_half((ushort)(max_bits & 0xffffu));
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
