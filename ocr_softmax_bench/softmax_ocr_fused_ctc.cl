// Fused OCR softmax + greedy CTC argmax (one work-item per timestep).
// Outputs per-frame token id and softmax prob at argmax (for prob_preds).
//
// argmax(softmax(logits)) == argmax(logits), so token id needs no full softmax row.
// At argmax index k*: softmax[k*] = exp(0)/sum_exp = 1/sum_exp.
//
// Launch: global_size = seqlen (e.g. 128)

__kernel void softmax_ocr_fused_ctc(
    __global const float* restrict logits,
    __global int* restrict token_ids,
    __global float* restrict max_probs,
    const int seqlen,
    const int char_size)
{
    const int j = get_global_id(0);
    if (j >= seqlen) {
        return;
    }

    const int offset = j * char_size;
    const __global float* row_in = logits + offset;

    int best_k = 0;
    float max_logit = row_in[0];
    for (int k = 1; k < char_size; ++k) {
        const float v = row_in[k];
        if (v > max_logit) {
            max_logit = v;
            best_k = k;
        }
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < char_size; ++k) {
        sum_exp += exp(row_in[k] - max_logit);
    }

    token_ids[j] = best_k;
    max_probs[j] = 1.0f / sum_exp;
}
