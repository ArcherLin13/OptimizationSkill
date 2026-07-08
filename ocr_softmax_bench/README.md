# OCR Softmax Benchmark

CPU benchmark mirroring an OCR softmax OpenCL kernel: one thread per timestep `j`, row layout `logits[j * char_size + k]`.

Default shape: `seqlen=128`, `char_size=9973`.

## Versions

| Variant | Passes | `exp()` per row |
|---------|--------|-----------------|
| **baseline** | 3 (max → sum → normalize) | 2× `char_size` |
| **optimized** | 2 (max → exp+sum → scale) | 1× `char_size` |

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
| `softmax_ocr_opt.cl` | Optimized OpenCL kernel (1× exp per element) |
| `softmax_bench.cpp` | C++ benchmark (baseline + optimized + correctness) |
| `softmax_bench.js` | Node.js host benchmark (no compiler needed) |
| `softmax_bench.py` | Python fallback |
| `scripts/build_ohos.ps1` | Cross-compile for OHOS arm64 |
| `scripts/run_device.ps1` | Push binary to device and run |
