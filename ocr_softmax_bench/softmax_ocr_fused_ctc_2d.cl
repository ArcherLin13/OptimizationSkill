// Fused 2D: opt_2d parallelism + CTC argmax output (no full probs buffer).
//
// argmax(logits) via phase-1 reduction; sum_exp in phase 2; max_prob = 1/sum_exp.
//
// Launch: global={seqlen, LOCAL_CHAR}  local={1, LOCAL_CHAR}
// Local mem: reduce_buf, size = LOCAL_CHAR * 2 * sizeof(float)
//   [0..LOCAL_CHAR-1] float values, [LOCAL_CHAR..2*LOCAL_CHAR-1] int indices (aliased)

#ifndef LOCAL_CHAR
#define LOCAL_CHAR 256
#endif

__kernel void softmax_ocr_fused_ctc_2d(
    __global const float* restrict logits,
    __global int* restrict token_ids,
    __global float* restrict max_probs,
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

    __local int* idx_buf = (__local int*)&reduce_buf[LOCAL_CHAR];

    const int offset = j * char_size;
    const __global float* row_in = logits + offset;

    // --- Phase 1: argmax(logits) via strided scan + tree reduce ---
    float lane_max = -INFINITY;
    int lane_idx = 0;
    for (int k = lid; k < char_size; k += lsize) {
        const float v = row_in[k];
        if (v > lane_max) {
            lane_max = v;
            lane_idx = k;
        }
    }
    reduce_buf[lid] = lane_max;
    idx_buf[lid] = lane_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            const float other = reduce_buf[lid + stride];
            if (other > reduce_buf[lid]) {
                reduce_buf[lid] = other;
                idx_buf[lid] = idx_buf[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float row_max = reduce_buf[0];
    const int best_k = idx_buf[0];

    // --- Phase 2: sum_exp only (no probs write) ---
    float lane_sum = 0.0f;
    for (int k = lid; k < char_size; k += lsize) {
        lane_sum += exp(row_in[k] - row_max);
    }
    reduce_buf[lid] = lane_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            reduce_buf[lid] += reduce_buf[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        token_ids[j] = best_k;
        max_probs[j] = 1.0f / reduce_buf[0];
    }
}
