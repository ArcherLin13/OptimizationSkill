// CPU benchmark mirroring OCR softmax OpenCL kernel (seqlen x char_size rows).
// Baseline: 3 passes, exp() twice per element (original kernel).
// Optimized: 2 passes, exp() once, then scale by 1/sum.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace {

constexpr int kSeqLen = 128;
constexpr int kCharSize = 9973;
constexpr int kWarmup = 5;
constexpr int kRuns = 30;

using ClockTicks = clock_t;

double msSince(ClockTicks t0) {
    return 1000.0 * static_cast<double>(std::clock() - t0) / static_cast<double>(CLOCKS_PER_SEC);
}

void softmax_baseline(const float* logits, float* probs, int seqlen, int char_size) {
    for (int j = 0; j < seqlen; ++j) {
        const int offset = j * char_size;
        const float* row = logits + offset;
        float* out = probs + offset;

        float max_logit = row[0];
        for (int k = 1; k < char_size; ++k) {
            const float v = row[k];
            if (v > max_logit) {
                max_logit = v;
            }
        }

        float sum_exp = 0.f;
        for (int k = 0; k < char_size; ++k) {
            sum_exp += std::exp(row[k] - max_logit);
        }

        for (int k = 0; k < char_size; ++k) {
            out[k] = std::exp(row[k] - max_logit) / sum_exp;
        }
    }
}

void softmax_optimized(const float* logits, float* probs, int seqlen, int char_size) {
    for (int j = 0; j < seqlen; ++j) {
        const int offset = j * char_size;
        const float* row = logits + offset;
        float* out = probs + offset;

        float max_logit = row[0];
        for (int k = 1; k < char_size; ++k) {
            max_logit = std::fmax(max_logit, row[k]);
        }

        float sum_exp = 0.f;
        for (int k = 0; k < char_size; ++k) {
            const float e = std::exp(row[k] - max_logit);
            out[k] = e;
            sum_exp += e;
        }

        const float inv = 1.f / sum_exp;
        for (int k = 0; k < char_size; ++k) {
            out[k] *= inv;
        }
    }
}

double maxAbsDiff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::fmax(m, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return m;
}

void fillRandomLogits(float* logits, size_t n, unsigned seed) {
    std::srand(seed);
    for (size_t i = 0; i < n; ++i) {
        logits[i] = static_cast<float>((std::rand() % 2000) - 1000) * 0.01f;
    }
}

template <typename Fn>
double benchMs(Fn fn, const float* logits, float* probs, int seqlen, int char_size) {
    for (int i = 0; i < kWarmup; ++i) {
        fn(logits, probs, seqlen, char_size);
    }
    const ClockTicks t0 = std::clock();
    for (int i = 0; i < kRuns; ++i) {
        fn(logits, probs, seqlen, char_size);
    }
    return msSince(t0) / kRuns;
}

}  // namespace

int main() {
    const size_t n = static_cast<size_t>(kSeqLen) * static_cast<size_t>(kCharSize);
    std::vector<float> logits(n);
    std::vector<float> probsA(n);
    std::vector<float> probsB(n);

    fillRandomLogits(logits.data(), n, 42u);

    softmax_baseline(logits.data(), probsA.data(), kSeqLen, kCharSize);
    softmax_optimized(logits.data(), probsB.data(), kSeqLen, kCharSize);

    const double diff = maxAbsDiff(probsA.data(), probsB.data(), n);
    const double baselineMs = benchMs(softmax_baseline, logits.data(), probsA.data(), kSeqLen, kCharSize);
    const double optimizedMs = benchMs(softmax_optimized, logits.data(), probsB.data(), kSeqLen, kCharSize);

    const long long expBaseline = 2LL * kSeqLen * kCharSize;
    const long long expOptimized = 1LL * kSeqLen * kCharSize;

    std::printf("OCR softmax CPU benchmark (mirrors OpenCL kernel logic)\n");
    std::printf("  seqlen=%d  char_size=%d  elements=%zu\n", kSeqLen, kCharSize, n);
    std::printf("  exp calls per kernel: baseline=%lld  optimized=%lld\n", expBaseline, expOptimized);
    std::printf("  correctness max|diff|=%.3e %s\n", diff, diff < 1e-5 ? "PASS" : "FAIL");
    std::printf("  baseline  avg=%.3f ms  (%.2f M exp/s)\n", baselineMs,
                expBaseline / (baselineMs * 1e3));
    std::printf("  optimized avg=%.3f ms  (%.2f M exp/s)\n", optimizedMs,
                expOptimized / (optimizedMs * 1e3));
    std::printf("  speedup=%.2fx\n", baselineMs / optimizedMs);

    return diff < 1e-5 ? 0 : 1;
}
