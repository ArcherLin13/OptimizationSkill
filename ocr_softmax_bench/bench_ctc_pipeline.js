// End-to-end benchmark: original (softmax → full probs → decodeText)
// vs fused (softmax+argmax kernel → light decodeText).
//
// Usage: node bench_ctc_pipeline.js

const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");
const koffi = require("koffi");
const {
  makeCharset,
  decodeTextFromProbs,
  decodeTextFromArgmax,
  decodeResultsEqual,
  softmaxArgmaxFused,
} = require("./ctc_decode");

const SEQLEN = 128;
const CHAR_SIZE = 9973;
const N = SEQLEN * CHAR_SIZE;
const WARMUP = 3;
const RUNS = 20;
const SEED = 42;
const BATCH = 1;

const CL_SUCCESS = 0;
const CL_MEM_READ_ONLY = 1 << 0;
const CL_MEM_WRITE_ONLY = 1 << 1;
const CL_DEVICE_TYPE_GPU = 1 << 2;
const CL_DEVICE_TYPE_CPU = 1 << 1;
const CL_DEVICE_NAME = 0x102b;

const ocl = koffi.load("OpenCL.dll");
const clGetPlatformIDs = ocl.func("int clGetPlatformIDs(uint32 n, void* plats, uint32* nplats)");
const clGetDeviceIDs = ocl.func(
  "int clGetDeviceIDs(void* plat, uint32 type, uint32 n, void* devs, uint32* ndevs)"
);
const clGetDeviceInfo = ocl.func("int clGetDeviceInfo(void* dev, uint32 p, size_t sz, void* v, size_t* rsz)");
const clCreateContext = ocl.func("void* clCreateContext(void* props, uint32 n, void* devs, void* a, void* b, int* err)");
const clCreateCommandQueue = ocl.func("void* clCreateCommandQueue(void* ctx, void* dev, uint64 props, int* err)");
const clCreateProgramWithSource = ocl.func(
  "void* clCreateProgramWithSource(void* ctx, uint32 n, void* src, void* lens, int* err)"
);
const clBuildProgram = ocl.func("int clBuildProgram(void* prog, uint32 n, void* devs, const char* opts, void* a, void* b)");
const clCreateKernel = ocl.func("void* clCreateKernel(void* prog, const char* name, int* err)");
const clSetKernelArg = ocl.func("int clSetKernelArg(void* k, uint32 i, size_t sz, void* val)");
const clCreateBuffer = ocl.func("void* clCreateBuffer(void* ctx, uint64 flags, size_t sz, void* host, int* err)");
const clEnqueueWriteBuffer = ocl.func(
  "int clEnqueueWriteBuffer(void* q, void* buf, uint32 block, size_t off, size_t sz, void* ptr, uint32 nw, void* ew, void* ev)"
);
const clEnqueueNDRangeKernel = ocl.func(
  "int clEnqueueNDRangeKernel(void* q, void* k, uint32 dim, void* off, void* global, void* local, uint32 nw, void* ew, void* ev)"
);
const clEnqueueReadBuffer = ocl.func(
  "int clEnqueueReadBuffer(void* q, void* buf, uint32 block, size_t off, size_t sz, void* ptr, uint32 nw, void* ew, void* ev)"
);
const clFinish = ocl.func("int clFinish(void* q)");
const clReleaseMemObject = ocl.func("int clReleaseMemObject(void* m)");
const clReleaseKernel = ocl.func("int clReleaseKernel(void* k)");
const clReleaseProgram = ocl.func("int clReleaseProgram(void* p)");
const clReleaseCommandQueue = ocl.func("int clReleaseCommandQueue(void* q)");
const clReleaseContext = ocl.func("int clReleaseContext(void* c)");

function check(err, msg) {
  if (err !== CL_SUCCESS) throw new Error(`${msg} (err=${err})`);
}

function readU32(buf) {
  return buf.readUInt32LE(0);
}

function ptrBuf(handle) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(handle), 0);
  return b;
}

function writeSizeArray(sizes) {
  const b = Buffer.alloc(8 * sizes.length);
  for (let i = 0; i < sizes.length; i++) b.writeBigUInt64LE(BigInt(sizes[i]), i * 8);
  return b;
}

function fillLogits(logits) {
  let s = SEED;
  for (let i = 0; i < logits.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    logits[i] = ((s / 0xffffffff) * 2 - 1) * 10;
  }
}

function softmaxOpt1d(logits, probs, seqlen, charSize) {
  for (let j = 0; j < seqlen; j++) {
    const off = j * charSize;
    let mx = logits[off];
    for (let k = 1; k < charSize; k++) mx = Math.max(mx, logits[off + k]);
    let sum = 0;
    for (let k = 0; k < charSize; k++) {
      const e = Math.exp(logits[off + k] - mx);
      probs[off + k] = e;
      sum += e;
    }
    const inv = 1 / sum;
    for (let k = 0; k < charSize; k++) probs[off + k] *= inv;
  }
}

function benchFn(warmup, runs, fn) {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < runs; i++) fn();
  return (performance.now() - t0) / runs;
}

function createOpenCL(deviceType) {
  const nplat = Buffer.alloc(4);
  check(clGetPlatformIDs(0, null, nplat), "nplat");
  const np = readU32(nplat);
  const plats = Buffer.alloc(8 * np);
  check(clGetPlatformIDs(np, plats, nplat), "plats");
  const plat = koffi.decode(plats.subarray(0, 8), "void*");

  const ndev = Buffer.alloc(4);
  check(clGetDeviceIDs(plat, deviceType, 0, null, ndev), "ndev");
  const devs = Buffer.alloc(8);
  check(clGetDeviceIDs(plat, deviceType, 1, devs, ndev), "dev");
  const dev = koffi.decode(devs, "void*");

  const nameBuf = Buffer.alloc(256);
  check(clGetDeviceInfo(dev, CL_DEVICE_NAME, 256, nameBuf, null), "name");
  const name = nameBuf.toString("utf8").replace(/\0.*/, "");

  const err = Buffer.alloc(4);
  const ctx = clCreateContext(null, 1, devs, null, null, err);
  check(readU32(err), "ctx");
  const queue = clCreateCommandQueue(ctx, dev, 0n, err);
  check(readU32(err), "queue");
  return { ctx, queue, dev, devs, name };
}

function buildProgram(ctx, devs) {
  const src =
    fs.readFileSync(path.join(__dirname, "softmax_ocr_opt.cl"), "utf8") +
    "\n" +
    fs.readFileSync(path.join(__dirname, "softmax_ocr_fused_ctc.cl"), "utf8");
  const srcBuf = Buffer.from(src + "\0");
  const strPtrs = Buffer.alloc(8);
  strPtrs.writeBigUInt64LE(koffi.address(srcBuf), 0);
  const lengths = Buffer.alloc(8);
  lengths.writeBigUInt64LE(BigInt(srcBuf.length - 1), 0);
  const err = Buffer.alloc(4);
  const prog = clCreateProgramWithSource(ctx, 1, strPtrs, lengths, err);
  check(readU32(err), "program");
  check(clBuildProgram(prog, 1, devs, "", null, null), "build");
  return prog;
}

function runOclSoftmax(env, prog, logits) {
  const { ctx, queue, devs } = env;
  const err = Buffer.alloc(4);
  const kern = clCreateKernel(prog, "softmax_ocr_opt", err);
  check(readU32(err), "softmax_ocr_opt");

  const logitsBuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, BigInt(N * 4), null, err);
  const probsBuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, BigInt(N * 4), null, err);
  check(readU32(err), "bufs");
  check(clEnqueueWriteBuffer(queue, logitsBuf, 1, 0n, BigInt(N * 4), logits.buffer, 0, null, null), "write");

  const seqlenBuf = Buffer.alloc(4);
  seqlenBuf.writeInt32LE(SEQLEN, 0);
  const charSizeBuf = Buffer.alloc(4);
  charSizeBuf.writeInt32LE(CHAR_SIZE, 0);
  check(clSetKernelArg(kern, 0, 8n, ptrBuf(logitsBuf)), "a0");
  check(clSetKernelArg(kern, 1, 8n, ptrBuf(probsBuf)), "a1");
  check(clSetKernelArg(kern, 2, 4n, seqlenBuf), "a2");
  check(clSetKernelArg(kern, 3, 4n, charSizeBuf), "a3");

  const g = writeSizeArray([SEQLEN]);
  for (let i = 0; i < WARMUP; i++) {
    check(clEnqueueNDRangeKernel(queue, kern, 1, null, g, null, 0, null, null), "wu");
    check(clFinish(queue), "fin");
  }
  const t0 = performance.now();
  for (let i = 0; i < RUNS; i++) {
    check(clEnqueueNDRangeKernel(queue, kern, 1, null, g, null, 0, null, null), "k");
    check(clFinish(queue), "fin");
  }
  const kernelMs = (performance.now() - t0) / RUNS;

  const probs = new Float32Array(N);
  const tRead0 = performance.now();
  for (let i = 0; i < RUNS; i++) {
    check(clEnqueueReadBuffer(queue, probsBuf, 1, 0n, BigInt(N * 4), probs.buffer, 0, null, null), "read");
  }
  const readMs = (performance.now() - tRead0) / RUNS;

  clReleaseKernel(kern);
  clReleaseMemObject(logitsBuf);
  clReleaseMemObject(probsBuf);
  return { kernelMs, readMs, probs };
}

function runOclFused(env, prog, logits) {
  const { ctx, queue } = env;
  const err = Buffer.alloc(4);
  const kern = clCreateKernel(prog, "softmax_ocr_fused_ctc", err);
  check(readU32(err), "softmax_ocr_fused_ctc");

  const logitsBuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, BigInt(N * 4), null, err);
  const idsBuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, BigInt(SEQLEN * 4), null, err);
  const maxBuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, BigInt(SEQLEN * 4), null, err);
  check(readU32(err), "bufs");
  check(clEnqueueWriteBuffer(queue, logitsBuf, 1, 0n, BigInt(N * 4), logits.buffer, 0, null, null), "write");

  const seqlenBuf = Buffer.alloc(4);
  seqlenBuf.writeInt32LE(SEQLEN, 0);
  const charSizeBuf = Buffer.alloc(4);
  charSizeBuf.writeInt32LE(CHAR_SIZE, 0);
  check(clSetKernelArg(kern, 0, 8n, ptrBuf(logitsBuf)), "a0");
  check(clSetKernelArg(kern, 1, 8n, ptrBuf(idsBuf)), "a1");
  check(clSetKernelArg(kern, 2, 8n, ptrBuf(maxBuf)), "a2");
  check(clSetKernelArg(kern, 3, 4n, seqlenBuf), "a3");
  check(clSetKernelArg(kern, 4, 4n, charSizeBuf), "a4");

  const g = writeSizeArray([SEQLEN]);
  for (let i = 0; i < WARMUP; i++) {
    check(clEnqueueNDRangeKernel(queue, kern, 1, null, g, null, 0, null, null), "wu");
    check(clFinish(queue), "fin");
  }
  const t0 = performance.now();
  for (let i = 0; i < RUNS; i++) {
    check(clEnqueueNDRangeKernel(queue, kern, 1, null, g, null, 0, null, null), "k");
    check(clFinish(queue), "fin");
  }
  const kernelMs = (performance.now() - t0) / RUNS;

  const tokenIds = new Int32Array(SEQLEN);
  const maxProbs = new Float32Array(SEQLEN);
  const tRead0 = performance.now();
  for (let i = 0; i < RUNS; i++) {
    check(clEnqueueReadBuffer(queue, idsBuf, 1, 0n, BigInt(SEQLEN * 4), tokenIds.buffer, 0, null, null), "r0");
    check(clEnqueueReadBuffer(queue, maxBuf, 1, 0n, BigInt(SEQLEN * 4), maxProbs.buffer, 0, null, null), "r1");
  }
  const readMs = (performance.now() - tRead0) / RUNS;

  clReleaseKernel(kern);
  clReleaseMemObject(logitsBuf);
  clReleaseMemObject(idsBuf);
  clReleaseMemObject(maxBuf);
  return { kernelMs, readMs, tokenIds, maxProbs };
}

// --- main ---
const logits = new Float32Array(N);
const probs = new Float32Array(N);
const tokenIds = new Int32Array(SEQLEN);
const maxProbs = new Float32Array(SEQLEN);
const charset = makeCharset(CHAR_SIZE);

fillLogits(logits);
softmaxOpt1d(logits, probs, SEQLEN, CHAR_SIZE);

const refDecode = decodeTextFromProbs(probs, SEQLEN, CHAR_SIZE, charset);
softmaxArgmaxFused(logits, tokenIds, maxProbs, SEQLEN, CHAR_SIZE);
const fusedDecode = decodeTextFromArgmax(tokenIds, maxProbs, SEQLEN, charset);
const decodeOk = decodeResultsEqual(refDecode, fusedDecode);

console.log("OCR CTC pipeline benchmark (softmax + decodetext)");
console.log(`  host: ${process.platform} ${process.arch}  node=${process.version}`);
console.log(`  seqlen=${SEQLEN}  char_size=${CHAR_SIZE}  batch=${BATCH}`);
console.log(`  probs buffer: ${((N * 4) / 1024 / 1024).toFixed(2)} MB`);
console.log(`  fused output: ${((SEQLEN * 8) / 1024).toFixed(2)} KB (token_ids + max_probs)`);
console.log(`  decode correctness: ${decodeOk ? "PASS" : "FAIL"}`);
console.log(`  ref text len=${refDecode.pred_texts.length}  sample="${refDecode.pred_texts.slice(0, 8).join("")}"`);
console.log("");

// --- CPU breakdown ---
const cpuSoftmaxMs = benchFn(WARMUP, RUNS, () => softmaxOpt1d(logits, probs, SEQLEN, CHAR_SIZE));
const cpuDecodeOrigMs = benchFn(WARMUP, RUNS, () =>
  decodeTextFromProbs(probs, SEQLEN, CHAR_SIZE, charset)
);
const cpuFusedKernelMs = benchFn(WARMUP, RUNS, () =>
  softmaxArgmaxFused(logits, tokenIds, maxProbs, SEQLEN, CHAR_SIZE)
);
const cpuDecodeFusedMs = benchFn(WARMUP, RUNS, () =>
  decodeTextFromArgmax(tokenIds, maxProbs, SEQLEN, charset)
);

const cpuOrigE2E = cpuSoftmaxMs + cpuDecodeOrigMs;
const cpuFusedE2E = cpuFusedKernelMs + cpuDecodeFusedMs;

console.log("=== CPU end-to-end (Node.js) ===");
console.log(`  original: softmax=${cpuSoftmaxMs.toFixed(3)} ms + decode=${cpuDecodeOrigMs.toFixed(3)} ms => ${cpuOrigE2E.toFixed(3)} ms`);
console.log(`  fused:    kernel=${cpuFusedKernelMs.toFixed(3)} ms + decode=${cpuDecodeFusedMs.toFixed(4)} ms => ${cpuFusedE2E.toFixed(3)} ms`);
console.log(`  speedup:  ${(cpuOrigE2E / cpuFusedE2E).toFixed(2)}x`);
console.log(`  decode alone: ${cpuDecodeOrigMs.toFixed(3)} ms -> ${cpuDecodeFusedMs.toFixed(4)} ms (${(cpuDecodeOrigMs / Math.max(cpuDecodeFusedMs, 1e-6)).toFixed(0)}x)`);
console.log("");

// --- OpenCL ---
for (const [kind, dtype] of [
  ["GPU", CL_DEVICE_TYPE_GPU],
  ["CPU", CL_DEVICE_TYPE_CPU],
]) {
  let env;
  try {
    env = createOpenCL(dtype);
  } catch (e) {
    console.log(`OpenCL ${kind} unavailable: ${e.message}\n`);
    continue;
  }

  console.log(`=== OpenCL ${kind}: ${env.name} ===`);
  const prog = buildProgram(env.ctx, env.devs);

  const oclSoft = runOclSoftmax(env, prog, logits);
  const decodeOrigMs = benchFn(WARMUP, RUNS, () =>
    decodeTextFromProbs(oclSoft.probs, SEQLEN, CHAR_SIZE, charset)
  );
  const origE2E = oclSoft.kernelMs + oclSoft.readMs + decodeOrigMs;

  const oclFused = runOclFused(env, prog, logits);
  const decodeFusedMs = benchFn(WARMUP, RUNS, () =>
    decodeTextFromArgmax(oclFused.tokenIds, oclFused.maxProbs, SEQLEN, charset)
  );
  const fusedE2E = oclFused.kernelMs + oclFused.readMs + decodeFusedMs;

  console.log(
    `  original: kernel=${oclSoft.kernelMs.toFixed(3)} ms + read_probs=${oclSoft.readMs.toFixed(3)} ms + decode=${decodeOrigMs.toFixed(3)} ms => ${origE2E.toFixed(3)} ms`
  );
  console.log(
    `  fused:    kernel=${oclFused.kernelMs.toFixed(3)} ms + read_argmax=${oclFused.readMs.toFixed(4)} ms + decode=${decodeFusedMs.toFixed(4)} ms => ${fusedE2E.toFixed(3)} ms`
  );
  console.log(`  speedup e2e: ${(origE2E / fusedE2E).toFixed(2)}x`);
  console.log(`  readback saved: ${(oclSoft.readMs - oclFused.readMs).toFixed(3)} ms (${((1 - oclFused.readMs / oclSoft.readMs) * 100).toFixed(0)}%)`);

  clReleaseProgram(prog);
  clReleaseCommandQueue(env.queue);
  clReleaseContext(env.ctx);
  console.log("");
}
