// Full benchmark on this host: CPU mirrors + real OpenCL (Intel GPU).
// seqlen=128, char_size=9973, fixed seed logits for all variants.

const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");
const koffi = require("koffi");

const SEQLEN = 128;
const CHAR_SIZE = 9973;
const N = SEQLEN * CHAR_SIZE;
const WARMUP = 3;
const RUNS = 20;
const LOCAL_CHARS = [128, 256, 512];
const SEED = 42;

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

const BASELINE_CL = `
__kernel void softmax_ocr_baseline(__global const float* logits, __global float* probs, const int seqlen, const int char_size) {
    int j = get_global_id(0);
    if (j >= seqlen) return;
    int offset = j * char_size;
    float max_logit = logits[offset];
    for (int k = 1; k < char_size; ++k) max_logit = fmax(max_logit, logits[offset + k]);
    float sum_exp = 0.f;
    for (int k = 0; k < char_size; ++k) sum_exp += exp(logits[offset + k] - max_logit);
    for (int k = 0; k < char_size; ++k) probs[offset + k] = exp(logits[offset + k] - max_logit) / sum_exp;
}`;

function check(err, msg) {
  if (err !== CL_SUCCESS) throw new Error(`${msg} (err=${err})`);
}

function readU32(buf) {
  return buf.readUInt32LE(0);
}

function writeSizeArray(sizes) {
  const b = Buffer.alloc(8 * sizes.length);
  for (let i = 0; i < sizes.length; i++) b.writeBigUInt64LE(BigInt(sizes[i]), i * 8);
  return b;
}

function ptrBuf(handle) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(handle), 0);
  return b;
}

function fillLogits(logits) {
  let s = SEED;
  for (let i = 0; i < logits.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    logits[i] = ((s / 0xffffffff) * 2 - 1) * 10;
  }
}

function treeReduceMax(buf, lsize) {
  for (let stride = lsize >> 1; stride > 0; stride >>= 1) {
    for (let lid = 0; lid < stride; lid++) buf[lid] = Math.max(buf[lid], buf[lid + stride]);
  }
  return buf[0];
}

function treeReduceSum(buf, lsize) {
  for (let stride = lsize >> 1; stride > 0; stride >>= 1) {
    for (let lid = 0; lid < stride; lid++) buf[lid] += buf[lid + stride];
  }
  return buf[0];
}

function softmaxBaseline(logits, probs) {
  for (let j = 0; j < SEQLEN; j++) {
    const off = j * CHAR_SIZE;
    let mx = logits[off];
    for (let k = 1; k < CHAR_SIZE; k++) mx = Math.max(mx, logits[off + k]);
    let sum = 0;
    for (let k = 0; k < CHAR_SIZE; k++) sum += Math.exp(logits[off + k] - mx);
    for (let k = 0; k < CHAR_SIZE; k++) probs[off + k] = Math.exp(logits[off + k] - mx) / sum;
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

function softmaxOpt2d(logits, probs, localChar) {
  const laneMax = new Float64Array(localChar);
  const laneSum = new Float64Array(localChar);
  for (let j = 0; j < SEQLEN; j++) {
    const off = j * CHAR_SIZE;
    laneMax.fill(-Infinity);
    for (let lid = 0; lid < localChar; lid++) {
      for (let k = lid; k < CHAR_SIZE; k += localChar) laneMax[lid] = Math.max(laneMax[lid], logits[off + k]);
    }
    const mx = treeReduceMax(laneMax, localChar);
    laneSum.fill(0);
    for (let lid = 0; lid < localChar; lid++) {
      for (let k = lid; k < CHAR_SIZE; k += localChar) {
        const e = Math.exp(logits[off + k] - mx);
        probs[off + k] = e;
        laneSum[lid] += e;
      }
    }
    const inv = 1 / treeReduceSum(laneSum, localChar);
    for (let lid = 0; lid < localChar; lid++) {
      for (let k = lid; k < CHAR_SIZE; k += localChar) probs[off + k] *= inv;
    }
  }
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function benchCpu(name, fn, logits, out, ref) {
  for (let i = 0; i < WARMUP; i++) fn(logits, out);
  const t0 = performance.now();
  for (let i = 0; i < RUNS; i++) fn(logits, out);
  return { name, ms: (performance.now() - t0) / RUNS, diff: maxAbsDiff(out, ref) };
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

function buildProgram(ctx, devs, options) {
  const src =
    BASELINE_CL +
    "\n" +
    fs.readFileSync(path.join(__dirname, "softmax_ocr_opt.cl"), "utf8") +
    "\n" +
    fs.readFileSync(path.join(__dirname, "softmax_ocr_opt_2d.cl"), "utf8");
  const srcBuf = Buffer.from(src + "\0");
  const strPtrs = Buffer.alloc(8);
  strPtrs.writeBigUInt64LE(koffi.address(srcBuf), 0);
  const lengths = Buffer.alloc(8);
  lengths.writeBigUInt64LE(BigInt(srcBuf.length - 1), 0);
  const err = Buffer.alloc(4);
  const prog = clCreateProgramWithSource(ctx, 1, strPtrs, lengths, err);
  check(readU32(err), "program");
  check(clBuildProgram(prog, 1, devs, options || "", null, null), `build opts=${options}`);
  return prog;
}

function runOclKernel(env, prog, kernelName, logits, global, local, localMemBytes) {
  const { ctx, queue } = env;
  const err = Buffer.alloc(4);
  const kern = clCreateKernel(prog, kernelName, err);
  check(readU32(err), kernelName);

  const logitsBuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, BigInt(N * 4), null, err);
  check(readU32(err), "logitsBuf");
  const probsBuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, BigInt(N * 4), null, err);
  check(readU32(err), "probsBuf");
  check(clEnqueueWriteBuffer(queue, logitsBuf, 1, 0n, BigInt(N * 4), logits.buffer, 0, null, null), "write");

  const seqlenBuf = Buffer.alloc(4);
  seqlenBuf.writeInt32LE(SEQLEN, 0);
  const charSizeBuf = Buffer.alloc(4);
  charSizeBuf.writeInt32LE(CHAR_SIZE, 0);
  check(clSetKernelArg(kern, 0, 8n, ptrBuf(logitsBuf)), "arg0");
  check(clSetKernelArg(kern, 1, 8n, ptrBuf(probsBuf)), "arg1");
  check(clSetKernelArg(kern, 2, 4n, seqlenBuf), "arg2");
  check(clSetKernelArg(kern, 3, 4n, charSizeBuf), "arg3");
  if (localMemBytes) check(clSetKernelArg(kern, 4, BigInt(localMemBytes), null), "local");

  const g = writeSizeArray(global);
  const l = local ? writeSizeArray(local) : null;

  for (let i = 0; i < WARMUP; i++) {
    check(clEnqueueNDRangeKernel(queue, kern, global.length, null, g, l, 0, null, null), "warmup");
    check(clFinish(queue), "finish");
  }

  const t0 = performance.now();
  for (let i = 0; i < RUNS; i++) {
    check(clEnqueueNDRangeKernel(queue, kern, global.length, null, g, l, 0, null, null), "kernel");
    check(clFinish(queue), "finish");
  }
  const ms = (performance.now() - t0) / RUNS;

  const out = new Float32Array(N);
  check(clEnqueueReadBuffer(queue, probsBuf, 1, 0n, BigInt(N * 4), out.buffer, 0, null, null), "read");

  let rowSumErr = 0;
  for (let j = 0; j < SEQLEN; j++) {
    let s = 0;
    const off = j * CHAR_SIZE;
    for (let k = 0; k < CHAR_SIZE; k++) s += out[off + k];
    if (Math.abs(s - 1) > 1e-3) rowSumErr++;
  }

  clReleaseKernel(kern);
  clReleaseMemObject(logitsBuf);
  clReleaseMemObject(probsBuf);
  return { ms, out, rowSumErr };
}

// --- main ---
const logits = new Float32Array(N);
const ref = new Float32Array(N);
const tmp = new Float32Array(N);
fillLogits(logits);
softmaxOpt1d(logits, ref);

console.log("OCR softmax full benchmark (this PC)");
console.log(`  seqlen=${SEQLEN}  char_size=${CHAR_SIZE}  seed=${SEED}`);
console.log(`  warmup=${WARMUP}  runs=${RUNS}`);
console.log("");

const cpuResults = [
  benchCpu("cpu_baseline", (l, o) => softmaxBaseline(l, o), logits, tmp, ref),
  benchCpu("cpu_opt_1d", (l, o) => softmaxOpt1d(l, o), logits, tmp, ref),
];
for (const lc of LOCAL_CHARS) {
  cpuResults.push(benchCpu(`cpu_opt_2d_lc${lc}`, (l, o) => softmaxOpt2d(l, o, lc), logits, tmp, ref));
}

console.log("=== CPU (Node.js, mirrors OpenCL logic) ===");
for (const r of cpuResults) {
  console.log(`  ${r.name.padEnd(22)} ${r.ms.toFixed(2).padStart(7)} ms  max|diff|=${r.diff.toExponential(3)}`);
}
console.log(`  speedup opt_1d vs baseline: ${(cpuResults[0].ms / cpuResults[1].ms).toFixed(2)}x`);
console.log("");

const oclResults = [];
function benchOcl(env, prog, label, kernel, global, local, localBytes) {
  try {
    const { ms, out, rowSumErr } = runOclKernel(env, prog, kernel, logits, global, local, localBytes);
    const diff = maxAbsDiff(out, ref);
    oclResults.push({ label, ms, diff });
    console.log(
      `  ${label.padEnd(28)} ${ms.toFixed(2).padStart(7)} ms  max|diff|=${diff.toExponential(3)} rows_bad=${rowSumErr} ${diff < 1e-3 && rowSumErr === 0 ? "PASS" : "FAIL"}`
    );
  } catch (e) {
    console.log(`  ${label.padEnd(28)} ERROR: ${e.message}`);
  }
}

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
  const prog256 = buildProgram(env.ctx, env.devs, "");
  benchOcl(env, prog256, "ocl_baseline_1d", "softmax_ocr_baseline", [SEQLEN], null, 0);
  benchOcl(env, prog256, "ocl_opt_1d", "softmax_ocr_opt", [SEQLEN], null, 0);
  for (const lc of LOCAL_CHARS) {
    const prog = lc === 256 ? prog256 : buildProgram(env.ctx, env.devs, `-DLOCAL_CHAR=${lc}`);
    benchOcl(env, prog, `ocl_opt_2d_lc${lc}`, "softmax_ocr_opt_2d", [SEQLEN, lc], [1, lc], lc * 4);
    if (prog !== prog256) clReleaseProgram(prog);
  }
  clReleaseProgram(prog256);

  const base = oclResults.find((r) => r.label === "ocl_baseline_1d");
  const oneD = oclResults.find((r) => r.label === "ocl_opt_1d");
  const best2d = oclResults
    .filter((r) => r.label.startsWith("ocl_opt_2d") && r.diff < 1e-3)
    .sort((a, b) => a.ms - b.ms)[0];
  if (base && oneD && best2d) {
    console.log(`  speedup ${best2d.label} vs baseline: ${(base.ms / best2d.ms).toFixed(2)}x`);
    console.log(`  speedup ${best2d.label} vs ocl_opt_1d: ${(oneD.ms / best2d.ms).toFixed(2)}x`);
  }
  clReleaseCommandQueue(env.queue);
  clReleaseContext(env.ctx);
  console.log("");
}
