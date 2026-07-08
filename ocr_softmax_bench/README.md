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

## Quick run (Windows / any host with Node.js)

```powershell
cd ocr_softmax_bench
node softmax_bench.js
```

Example result on win32 x64 (Node v24):

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

## Files

| File | Purpose |
|------|---------|
| `softmax_ocr_opt.cl` | 1D optimized kernel (1× exp, 1 thread/row) |
| `softmax_ocr_opt_2d.cl` | 2D optimized kernel (`gy` parallel + local reduce) |
| `softmax_bench.cpp` | C++ benchmark (baseline + optimized + correctness) |
| `softmax_bench.js` | Node.js host benchmark (no compiler needed) |
| `softmax_bench.py` | Python fallback |
| `scripts/build_ohos.ps1` | Cross-compile for OHOS arm64 |
| `scripts/run_device.ps1` | Push binary to device and run |
