# Dual NPU model: serial vs parallel inference

Compare two MindSpore models on device (NNRT / HiSilicon NPU):

| Mode | What happens |
|------|----------------|
| **serial** | `predict(A)` then `predict(B)` |
| **parallel** | `predict(A)` and `predict(B)` on two threads at once |

Models are **loaded once** before timing; benchmark is **infer only**.

## Run

```powershell
cd hisi_model_bench
.\scripts\download_model.ps1    # mobilenetv2.ms (~14 MB)
.\scripts\build_ohos.ps1
.\scripts\run_device_dual.ps1
```

若 NNRT 仍失败，先试 CPU：

```powershell
.\scripts\run_device_dual.ps1 -Device cpu
```

Two different models:

```powershell
.\scripts\run_device_dual.ps1 -ModelA "testdata\model_a.ms" -ModelB "testdata\model_b.ms"
```

Same model twice (default, two independent instances):

```powershell
.\scripts\run_device_dual.ps1 -ModelA "testdata\tiny.ms"
```

## Read results

```text
serial wall:    X ms  (A=.. + B=..)
parallel wall:  Y ms  (A=.., B=.. concurrent)
speedup: X/Y
```

| If parallel wall is… | Meaning |
|----------------------|---------|
| ~ `A + B` (same as serial) | NPU **serializes** both models |
| ~ `max(A, B)` | NPU **allows parallel** execution |
| in between | partial overlap |

## Files

- `device_test/ms_dual_infer_bench.cpp` — benchmark binary `ms_dual_bench`
- `device_test/ms_common.h` — shared MindSpore load/predict helpers
- `scripts/run_device_dual.ps1` — push + run on phone
