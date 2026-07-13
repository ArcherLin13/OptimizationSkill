// Generate OCR softmax test vectors for device kernel bring-up.
// Usage: node generate_testdata.js [out_dir]
//
// Outputs:
//   logits.bin       float32[seqlen * char_size]  — kernel input
//   probs_ref.bin    float32[seqlen * char_size]  — golden output (opt 1D, same math as 2D)
//   manifest.json    dims, launch config, file sizes (NOT reduce_buf — see below)

const fs = require("fs");
const path = require("path");

const SEQLEN = 128;
const CHAR_SIZE = 9973;
const SEED = 42;
const LOCAL_CHAR = 512; // recommended 2D launch; change if your device uses 256

const N = SEQLEN * CHAR_SIZE;

function fillLogits(logits) {
  let s = SEED;
  for (let i = 0; i < logits.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    logits[i] = ((s / 0xffffffff) * 2 - 1) * 10;
  }
}

function softmaxOpt1d(logits, probs) {
  for (let j = 0; j < SEQLEN; j++) {
    const off = j * CHAR_SIZE;
    let mx = logits[off];
    for (let k = 1; k < CHAR_SIZE; k++) mx = Math.max(mx, logits[off + k]);
    let sum = 0;
    for (let k = 0; k < CHAR_SIZE; k++) {
      const e = Math.exp(logits[off + k] - mx);
      probs[off + k] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let k = 0; k < CHAR_SIZE; k++) probs[off + k] *= inv;
  }
}

const outDir = process.argv[2] || path.join(__dirname, "testdata");
fs.mkdirSync(outDir, { recursive: true });

const logits = new Float32Array(N);
const probs = new Float32Array(N);
fillLogits(logits);
softmaxOpt1d(logits, probs);

const logitsPath = path.join(outDir, "logits.bin");
const probsPath = path.join(outDir, "probs_ref.bin");
fs.writeFileSync(logitsPath, Buffer.from(logits.buffer));
fs.writeFileSync(probsPath, Buffer.from(probs.buffer));

const manifest = {
  description: "OCR softmax test vectors (seqlen x char_size row-major)",
  seqlen: SEQLEN,
  char_size: CHAR_SIZE,
  num_elements: N,
  dtype: "float32",
  seed: SEED,
  files: {
    logits: {
      path: "logits.bin",
      bytes: N * 4,
      layout: "logits[j * char_size + k]",
    },
    probs_ref: {
      path: "probs_ref.bin",
      bytes: N * 4,
      layout: "expected softmax output",
    },
  },
  kernels: {
    opt_1d: {
      file: "softmax_ocr_opt.cl",
      entry: "softmax_ocr_opt",
      global_size: [SEQLEN],
      local_size: [0],
      local_mem_bytes: 0,
    },
    opt_2d: {
      file: "softmax_ocr_opt_2d.cl",
      entry: "softmax_ocr_opt_2d",
      global_size: [SEQLEN, LOCAL_CHAR],
      local_size: [1, LOCAL_CHAR],
      local_mem_bytes: LOCAL_CHAR * 4,
      local_char: LOCAL_CHAR,
      local_mem_arg: {
        name: "reduce_buf",
        index: 4,
        note: "GPU scratch only — do NOT load from host binary",
      },
    },
  },
  reduce_buf: {
    is_host_input: false,
    size_bytes_formula: "LOCAL_CHAR * sizeof(float)",
    size_bytes_for_recommended_launch: LOCAL_CHAR * 4,
    depends_on: "local_size[1] (gy), must equal LOCAL_CHAR compile flag",
    fixed_for_given_launch: true,
  },
  verify: {
    compare_output_to: "probs_ref.bin",
    tolerance_max_abs_diff: 1e-3,
    row_sum_tolerance: 1e-3,
  },
};

fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));

console.log(`Wrote testdata to ${outDir}`);
console.log(`  logits.bin     ${N * 4} bytes`);
console.log(`  probs_ref.bin  ${N * 4} bytes`);
console.log(`  manifest.json`);
console.log("");
console.log("reduce_buf: NOT a file — allocate on device:");
console.log(`  clSetKernelArg(kernel, 4, ${LOCAL_CHAR * 4}, nullptr);`);
