#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ctc {

constexpr int kBlankIdx = 0;

struct DecodeResult {
    std::vector<float> prob_preds;
    std::vector<int> loc_preds;
    std::vector<int> token_preds;  // charset index per emitted char
};

inline void findMaxProbability(const float* probs_row, int char_size, int* out_k, float* out_p) {
    int best_k = 0;
    float best_p = probs_row[0];
    for (int k = 1; k < char_size; ++k) {
        const float p = probs_row[k];
        if (p > best_p) {
            best_p = p;
            best_k = k;
        }
    }
    *out_k = best_k;
    *out_p = best_p;
}

inline DecodeResult decodeTextFromProbs(const float* probs, int seqlen, int char_size) {
    DecodeResult out;
    int last_p = -1;
    for (int t = 0; t < seqlen; ++t) {
        const float* row = probs + static_cast<size_t>(t) * char_size;
        int p = 0;
        float max_prob = 0.f;
        findMaxProbability(row, char_size, &p, &max_prob);
        if (p != last_p && p != kBlankIdx) {
            out.prob_preds.push_back(max_prob);
            out.loc_preds.push_back(t);
            out.token_preds.push_back(p);
        }
        last_p = p;
    }
    return out;
}

inline DecodeResult decodeTextFromArgmax(const int* token_ids, const float* max_probs, int seqlen) {
    DecodeResult out;
    int last_p = -1;
    for (int t = 0; t < seqlen; ++t) {
        const int p = token_ids[t];
        const float max_prob = max_probs[t];
        if (p != last_p && p != kBlankIdx) {
            out.prob_preds.push_back(max_prob);
            out.loc_preds.push_back(t);
            out.token_preds.push_back(p);
        }
        last_p = p;
    }
    return out;
}

inline bool decodeResultsEqual(const DecodeResult& a, const DecodeResult& b, float prob_tol = 1e-4f) {
    if (a.prob_preds.size() != b.prob_preds.size()) {
        return false;
    }
    for (size_t i = 0; i < a.prob_preds.size(); ++i) {
        if (a.loc_preds[i] != b.loc_preds[i]) {
            return false;
        }
        if (a.token_preds[i] != b.token_preds[i]) {
            return false;
        }
        if (std::fabs(a.prob_preds[i] - b.prob_preds[i]) > prob_tol) {
            return false;
        }
    }
    return true;
}

}  // namespace ctc
