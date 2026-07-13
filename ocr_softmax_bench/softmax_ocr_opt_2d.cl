// OCR row softmax — 2D parallel variant.
// Layout: logits/probs [seqlen][char_size], row-major.
// Default: seqlen=128, char_size=9973.
//
// gx (dim 0): one work-group per timestep j  → get_group_id(0) == j
// gy (dim 1): LOCAL_CHAR lanes cooperate on the row via strided loops + local reduction
//
// vs softmax_ocr_opt (1D): same math (1× exp), but char_size work is split across gy threads.
//
// Launch: global={128, 512}  local={1, 512}
// Local mem: reduce_buf = 512 * sizeof(float) = 2048 B

#define LOCAL_CHAR 512

__kernel void softmax_ocr_opt_2d(
    __global const float* restrict logits,
    __global float* restrict probs,
    const int seqlen,
    const int char_size,
    __local float* restrict reduce_buf)
{
    const int j = get_group_id(0);
    const int lid = get_local_id(1);
    const int lsize = get_local_size(1);

    if (j >= seqlen) {
        return;
    }

    const int offset = j * char_size;
    const __global float* row_in = logits + offset;
    __global float* row_out = probs + offset;

    // --- Phase 1: row max (parallel strided scan + tree reduce in local mem) ---
    float lane_max = -INFINITY;
    for (int k = lid; k < char_size; k += lsize) {
        lane_max = fmax(lane_max, row_in[k]);
    }
    reduce_buf[lid] = lane_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            reduce_buf[lid] = fmax(reduce_buf[lid], reduce_buf[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = reduce_buf[0];

    // --- Phase 2: exp once per element, lane partial sum, reduce sum ---
    float lane_sum = 0.0f;
    for (int k = lid; k < char_size; k += lsize) {
        const float e = exp(row_in[k] - row_max);
        row_out[k] = e;
        lane_sum += e;
    }
    reduce_buf[lid] = lane_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            reduce_buf[lid] += reduce_buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inv_sum = 1.0f / reduce_buf[0];

    // --- Phase 3: normalize (no exp) ---
    for (int k = lid; k < char_size; k += lsize) {
        row_out[k] *= inv_sum;
    }
}

// Host setup example (C / OpenCL C++):
//
//   const size_t local[2]  = { 1, 256 };
//   const size_t global[2] = { (size_t)seqlen, local[1] };
//   clSetKernelArg(k, ..., sizeof(int), &seqlen);
//   clSetKernelArg(k, ..., sizeof(int), &char_size);
//   clSetKernelArg(k, ..., local[1] * sizeof(float), nullptr);  // reduce_buf
//   clEnqueueNDRangeKernel(..., 2, nullptr, global, local, ...);
