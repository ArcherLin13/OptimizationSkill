// OCR row softmax: one work-item per timestep j.
// Layout: logits/probs [seqlen][char_size], row-major.
// Default: seqlen=128, char_size=9973.
//
// Optimized vs baseline: compute exp() once per element, then scale by 1/sum.
// Launch: global_size = seqlen, local_size = 0 (runtime default).

__kernel void softmax_ocr_opt(
    __global const float* restrict logits,
    __global float* restrict probs,
    const int seqlen,
    const int char_size)
{
    const int j = get_global_id(0);
    if (j >= seqlen) {
        return;
    }

    const int offset = j * char_size;
    const __global float* row_in = logits + offset;
    __global float* row_out = probs + offset;

    // Pass 1: row max for numerical stability
    float max_logit = row_in[0];
    for (int k = 1; k < char_size; ++k) {
        max_logit = fmax(max_logit, row_in[k]);
    }

    // Pass 2: exp once, store unnormalized values and accumulate sum
    float sum_exp = 0.0f;
    for (int k = 0; k < char_size; ++k) {
        const float e = exp(row_in[k] - max_logit);
        row_out[k] = e;
        sum_exp += e;
    }

    // Pass 3: normalize (no exp)
    const float inv_sum = 1.0f / sum_exp;
    for (int k = 0; k < char_size; ++k) {
        row_out[k] *= inv_sum;
    }
}
