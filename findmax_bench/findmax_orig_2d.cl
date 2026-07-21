// Original-style findMaxValue (2D NDRange), same logic as described:
//   lid = lid_y * group_size_x + lid_x
//   __local float s_max[256]
//   load half -> float local
//   workgroup tree reduction
//   lid==0 updates global max
//
// Safety (only what is required to not crash/hang; logic unchanged):
//   - no early return before barrier (that deadlocks when local_size > 256)
//   - OOB gids load -inf instead of out-of-range read
//   - require local_size <= 256 (host should use 16x16)
//   - global max via atomic (multi-WG safe)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

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
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int group_size_x = get_local_size(0);
    const int group_size_y = get_local_size(1);
    const int local_size = group_size_x * group_size_y;
    const int lid = lid_y * group_size_x + lid_x;

    __local float s_max[256];

    float val = -65504.0f;
    if ((unsigned int)gid_x < width && (unsigned int)gid_y < height) {
        val = (float)src[(unsigned int)gid_y * width + (unsigned int)gid_x];
    }
    // Same intent as "if (lid > 256) return" without skipping barrier:
    // only lanes with lid < 256 participate in s_max (require local_size <= 256).
    if (lid < 256) {
        s_max[lid] = val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = local_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            s_max[lid] = max(s_max[lid], s_max[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const half local_res = (half)s_max[0];
        atomic_max_half(max_value, local_res);
    }
}
