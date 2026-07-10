# OCR Softmax Benchmark

CPU benchmark mirroring an OCR softmax OpenCL kernel: one thread per timestep `j`, row layout `logits[j * char_size + k]`.

Default shape: `seqlen=128`, `char_size=9973`.

## Versions

| Variant | Parallelism | `exp()` per row |
|---------|-------------|-----------------|
| **baseline** | 1D: 1 thread/row, serial over `char_size` | 2× `char_size` |
| **softmax_ocr_opt** | 1D: `global = {seqlen}` | 1× `char_size` |
| **softmax_ocr_opt_2d** | 2D: `global = {seqlen, LOCAL_CHAR}` | 1× `char_size`, split across `gy` |

### 1D launch (`softmax_ocr_opt.cl`)

```cpp
size_t global = seqlen;   // 128
size_t local  = 0;        // runtime default
// 1D: clEnqueueNDRangeKernel(..., 1, nullptr, &global, nullptr, ...)
```

### 2D launch (`softmax_ocr_opt_2d.cl`) — recommended on GPU

```cpp
const size_t local[2]  = { 1, 256 };              // gy = 256 lanes per row
const size_t global[2] = { seqlen, local[1] };    // { 128, 256 }
// local mem: LOCAL_CHAR * sizeof(float) for reduce_buf
// 2D: clEnqueueNDRangeKernel(..., 2, nullptr, global, local, ...)
```

Each row is one work-group (`gx = j`). `gy` threads stride over `char_size` (~39 elements/lane for 9973÷256), then tree-reduce max/sum in `__local` memory. Tune `LOCAL_CHAR` to 128 / 256 / 512 for your GPU.

## Test vectors (for your device)

```powershell
node generate_testdata.js
# → testdata/logits.bin, probs_ref.bin, manifest.json
```

| File | Size | Role |
|------|------|------|
| `logits.bin` | 128×9973×4 = **5,106,176 B** | kernel input |
| `probs_ref.bin` | same | golden output (compare your kernel) |
| `manifest.json` | — | dims, `global_size`, `local_mem` |

**`reduce_buf` is not a binary file.** It is on-chip local memory sized at launch:

```
bytes = LOCAL_CHAR × 4   (512 → 2048 B, 256 → 1024 B)
clSetKernelArg(kernel, 4, bytes, nullptr);
```

Fixed for a given launch: tied to `local_size[1]` / `-DLOCAL_CHAR=`, not `char_size`.

```powershell
cd ocr_softmax_bench
npm install koffi    # once, for real OpenCL GPU benchmark
node run_bench.js    # CPU mirrors + Intel/AMD/NVIDIA OpenCL if available
node softmax_bench.js
```

Example result on win32 x64 (Node v24, Intel Iris Xe GPU):

```
OpenCL GPU:
  ocl_baseline_1d     ~16 ms
  ocl_opt_1d          ~15 ms   (matches ~17 ms on device)
  ocl_opt_2d_lc256    ~0.3 ms  (gy parallel + local reduce)
```

CPU-only (`softmax_bench.js`):

```
baseline  avg=30.06 ms
optimized avg=15.33 ms
speedup=1.96x
correctness PASS
```

## C++ (host or cross-compile)

```powershell
# OHOS arm64 (DevEco SDK)
.\scripts\build_ohos.ps1
.\scripts\run_device.ps1   # push + run on device via hdc
```

Native host: compile `softmax_bench.cpp` with any C++17 compiler.

## CTC fused pipeline (softmax + argmax → decodetext)

Greedy CTC decode after softmax:

```text
original:  softmax → probs[128×9973] → CPU findMaxProbability × 128 → CTC collapse
fused:     softmax_ocr_fused_ctc → token_ids[128] + max_probs[128] → CPU decodeTextFromArgmax
```

```powershell
node bench_ctc_pipeline.js       # PC (OpenCL + CPU)
.\scripts\run_device_ctc.ps1   # HarmonyOS phone via hdc
```

See `HARMONYOS_CTC_TEST.md` for device details.

| File | Purpose |
|------|---------|
| `softmax_ocr_fused_ctc.cl` | Fused kernel (argmax on logits + 1× exp for prob at argmax) |
| `ctc_decode.js` | CPU `decodeTextFromProbs` / `decodeTextFromArgmax` (blank=0) |
| `bench_ctc_pipeline.js` | End-to-end timing: original vs fused |

## Files

| File | Purpose |
|------|---------|
| `softmax_ocr_opt.cl` | 1D optimized kernel (1× exp, 1 thread/row) |
| `softmax_ocr_opt_2d.cl` | 2D optimized kernel (`gy` parallel + local reduce) |
| `softmax_ocr_fused_ctc.cl` | Fused softmax+argmax for CTC greedy decode |
| `ctc_decode.js` | CPU CTC decodetext helpers |
| `bench_ctc_pipeline.js` | Original vs fused pipeline benchmark |
| `softmax_bench.cpp` | C++ benchmark (baseline + optimized + correctness) |
| `run_bench.js` | CPU + real OpenCL benchmark (needs `npm install koffi`) |
| `softmax_bench.py` | Python fallback |
| `scripts/build_ohos.ps1` | Cross-compile for OHOS arm64 |
| `scripts/run_device.ps1` | Push binary to device and run |
