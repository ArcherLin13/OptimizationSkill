// OCR softmax benchmark: baseline (2x exp) vs optimized (1x exp)
// seqlen=128, char_size=9973

const SEQLEN = 128;
const CHAR_SIZE = 9973;
const WARMUP = 5;
const RUNS = 30;

function softmaxBaseline(logits, probs, seqlen, charSize) {
  for (let j = 0; j < seqlen; j++) {
    const offset = j * charSize;
    let maxLogit = logits[offset];
    for (let k = 1; k < charSize; k++) {
      const v = logits[offset + k];
      if (v > maxLogit) maxLogit = v;
    }
    let sumExp = 0;
    for (let k = 0; k < charSize; k++) {
      sumExp += Math.exp(logits[offset + k] - maxLogit);
    }
    for (let k = 0; k < charSize; k++) {
      probs[offset + k] = Math.exp(logits[offset + k] - maxLogit) / sumExp;
    }
  }
}

function softmaxOptimized(logits, probs, seqlen, charSize) {
  for (let j = 0; j < seqlen; j++) {
    const offset = j * charSize;
    let maxLogit = logits[offset];
    for (let k = 1; k < charSize; k++) {
      if (logits[offset + k] > maxLogit) maxLogit = logits[offset + k];
    }
    let sumExp = 0;
    for (let k = 0; k < charSize; k++) {
      const e = Math.exp(logits[offset + k] - maxLogit);
      probs[offset + k] = e;
      sumExp += e;
    }
    const inv = 1 / sumExp;
    for (let k = 0; k < charSize; k++) {
      probs[offset + k] *= inv;
    }
  }
}

function bench(fn, logits, probs) {
  for (let i = 0; i < WARMUP; i++) fn(logits, probs, SEQLEN, CHAR_SIZE);
  const t0 = performance.now();
  for (let i = 0; i < RUNS; i++) fn(logits, probs, SEQLEN, CHAR_SIZE);
  return (performance.now() - t0) / RUNS;
}

const n = SEQLEN * CHAR_SIZE;
const logits = new Float64Array(n);
const probsA = new Float64Array(n);
const probsB = new Float64Array(n);

for (let i = 0; i < n; i++) logits[i] = (Math.random() * 2 - 1) * 10;

softmaxBaseline(logits, probsA, SEQLEN, CHAR_SIZE);
softmaxOptimized(logits, probsB, SEQLEN, CHAR_SIZE);

let diff = 0;
for (let i = 0; i < n; i++) diff = Math.max(diff, Math.abs(probsA[i] - probsB[i]));

const baseMs = bench(softmaxBaseline, logits, probsA);
const optMs = bench(softmaxOptimized, logits, probsB);
const expBase = 2 * SEQLEN * CHAR_SIZE;
const expOpt = SEQLEN * CHAR_SIZE;

console.log("OCR softmax benchmark (Node.js, same logic as your OpenCL kernel)");
console.log(`  host: ${process.platform} ${process.arch}  node=${process.version}`);
console.log(`  seqlen=${SEQLEN}  char_size=${CHAR_SIZE}  elements=${n}`);
console.log(`  exp calls: baseline=${expBase}  optimized=${expOpt}`);
console.log(`  correctness max|diff|=${diff.toExponential(3)} ${diff < 1e-10 ? "PASS" : "FAIL"}`);
console.log(`  baseline  avg=${baseMs.toFixed(2)} ms`);
console.log(`  optimized avg=${optMs.toFixed(2)} ms`);
console.log(`  speedup=${(baseMs / optMs).toFixed(2)}x`);
console.log(`  est. GPU kernel savings: ~${((1 - expOpt / expBase) * 100).toFixed(0)}% less exp() work`);
