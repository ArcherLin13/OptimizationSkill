// Baseline findMaxValue:
//   each WI loads pixel(s) -> local reduction in workgroup -> atomic update global max.
// API: (__global half* src, unsigned int width, unsigned int height, __global half* max_value)
//
// Requires: cl_khr_fp16, 32-bit global atomics (cmpxchg).
// max_value buffer must be >= 4 bytes and 4-byte aligned (half stored in low 16 bits).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef WG_SIZE
#define WG_SIZE 256
#endif

inline void atomic_max_half(__global half* addr, half val) {
    // Treat first 4 bytes as uint; half lives in low 16 bits.
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
    __local half lmax[WG_SIZE];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);
    const uint n = width * height;

    half v = (gid < n) ? src[gid] : (half)(-65504.0f);
    lmax[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            lmax[lid] = max(lmax[lid], lmax[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_max_half(max_value, lmax[0]);
    }
}
