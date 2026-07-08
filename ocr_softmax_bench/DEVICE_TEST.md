# Device test: softmax_ocr_opt_2d

Test **correctness** and **OpenCL profiling time** on your device.

## 1. Prepare testdata (PC)

```powershell
cd ocr_softmax_bench
node generate_testdata.js
# optional placeholder for arg5:
python generate_before_arg_5.py
```

Files:

| File | Role |
|------|------|
| `testdata/logits.bin` | input |
| `testdata/probs_ref.bin` | golden output |
| `softmax_ocr_opt_2d.cl` | kernel |

## 2. Launch config (2D)

```text
kernel:     softmax_ocr_opt_2d
build:      -DLOCAL_CHAR=512
global:     { 128, 512 }
local:      {   1, 512 }
arg0 logits   global buffer  5106176 B
arg1 probs    global buffer  5106176 B
arg2 seqlen   int            128
arg3 char_size int           9973
arg4 reduce_buf  local only  2048 B  (clSetKernelArg(..., nullptr))
```

If 512 fails on device, try `LOCAL_CHAR=256`, `global/local gy=256`, local mem 1024 B.

## 3. Correctness

Compare `probs` output vs `probs_ref.bin`:

- `max|diff| < 1e-3` → PASS
- each row sum ≈ 1.0 (tolerance 1e-3)

## 4. OpenCL profiling time

Requirements:

1. Queue: `CL_QUEUE_PROFILING_ENABLE`
2. Pass non-null `cl_event` to `clEnqueueNDRangeKernel`
3. After `clWaitForEvents`:

```cpp
cl_ulong t0, t1;
clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
double ms = (t1 - t0) / 1e6;   // kernel GPU time only
```

Do **not** mix with `clFinish` wall time unless you want end-to-end.

Warmup 3 + average 20 runs is recommended.

## 5. Reference host binary

`device_test/ocl_test_2d.cpp` — compile on any platform with OpenCL:

```bash
g++ -O2 -std=c++17 device_test/ocl_test_2d.cpp -lOpenCL -o ocl_test_2d
./ocl_test_2d --data testdata --local-char 512 --runs 20
```

Example output:

```text
correctness max|diff|=...  rows_bad=0  PASS
OpenCL profiling (kernel only): avg=XX.XXX ms
```

## 6. Push to HarmonyOS device (if using hdc)

```powershell
.\scripts\run_device_2d.ps1
```

Pushes: `ocl_test_2d`, `softmax_ocr_opt_2d.cl`, `testdata/*.bin`

## 7. Your own test platform

If you already run arbitrary kernels:

1. Load `logits.bin` → input buffer  
2. Run `softmax_ocr_opt_2d` with table above  
3. Read `probs` → compare to `probs_ref.bin`  
4. Use event profiling on the enqueue call  

`before_arg_5.bin` is optional (only if your harness requires a file per arg); kernel ignores initial local mem content.
