// Original OCR softmax: 3 passes, exp() twice per element.

__kernel void softmax_ocr_baseline(
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

    float max_logit = row_in[0];
    for (int k = 1; k < char_size; ++k) {
        max_logit = fmax(max_logit, row_in[k]);
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < char_size; ++k) {
        sum_exp += exp(row_in[k] - max_logit);
    }

    for (int k = 0; k < char_size; ++k) {
        row_out[k] = exp(row_in[k] - max_logit) / sum_exp;
    }
}
