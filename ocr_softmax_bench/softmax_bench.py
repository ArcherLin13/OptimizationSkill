#!/usr/bin/env python3
"""CPU benchmark: OCR softmax baseline (2x exp) vs optimized (1x exp)."""

import math
import random
import time

SEQLEN = 128
CHAR_SIZE = 9973
WARMUP = 3
RUNS = 15


def softmax_baseline(logits, probs, seqlen, char_size):
    for j in range(seqlen):
        offset = j * char_size
        row = logits[offset : offset + char_size]
        max_logit = row[0]
        for k in range(1, char_size):
            v = row[k]
            if v > max_logit:
                max_logit = v
        sum_exp = 0.0
        for k in range(char_size):
            sum_exp += math.exp(row[k] - max_logit)
        for k in range(char_size):
            probs[offset + k] = math.exp(row[k] - max_logit) / sum_exp


def softmax_optimized(logits, probs, seqlen, char_size):
    for j in range(seqlen):
        offset = j * char_size
        row = logits[offset : offset + char_size]
        max_logit = row[0]
        for k in range(1, char_size):
            if row[k] > max_logit:
                max_logit = row[k]
        sum_exp = 0.0
        for k in range(char_size):
            e = math.exp(row[k] - max_logit)
            probs[offset + k] = e
            sum_exp += e
        inv = 1.0 / sum_exp
        for k in range(char_size):
            probs[offset + k] *= inv


def bench(fn, logits, probs):
    for _ in range(WARMUP):
        fn(logits, probs, SEQLEN, CHAR_SIZE)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        fn(logits, probs, SEQLEN, CHAR_SIZE)
    return (time.perf_counter() - t0) * 1000.0 / RUNS


def main():
    n = SEQLEN * CHAR_SIZE
    random.seed(42)
    logits = [random.uniform(-10.0, 10.0) for _ in range(n)]
    probs_a = [0.0] * n
    probs_b = [0.0] * n

    softmax_baseline(logits, probs_a, SEQLEN, CHAR_SIZE)
    softmax_optimized(logits, probs_b, SEQLEN, CHAR_SIZE)
    diff = max(abs(a - b) for a, b in zip(probs_a, probs_b))

    base_ms = bench(softmax_baseline, logits, probs_a)
    opt_ms = bench(softmax_optimized, logits, probs_b)
    exp_base = 2 * SEQLEN * CHAR_SIZE
    exp_opt = SEQLEN * CHAR_SIZE

    print("OCR softmax CPU benchmark (Python math.exp, same logic as OpenCL kernel)")
    print(f"  seqlen={SEQLEN}  char_size={CHAR_SIZE}  elements={n}")
    print(f"  exp calls: baseline={exp_base}  optimized={exp_opt}")
    print(f"  correctness max|diff|={diff:.3e} {'PASS' if diff < 1e-5 else 'FAIL'}")
    print(f"  baseline  avg={base_ms:.2f} ms")
    print(f"  optimized avg={opt_ms:.2f} ms")
    print(f"  speedup={base_ms / opt_ms:.2f}x")
    print()
    print("Note: Python is slower than native/GPU; speedup ratio is what matters here.")


if __name__ == "__main__":
    main()
