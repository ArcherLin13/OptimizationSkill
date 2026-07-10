# HarmonyOS CTC pipeline device test

Compare on phone (**production baseline = opt_2d**, not 1D):

| Path | GPU | CPU |
|------|-----|-----|
| **opt_2d+decode** | `softmax_ocr_opt_2d` (global={128,512}) → read `probs` (4.87 MB) | `decodeTextFromProbs` |
| **fused+decode** | `softmax_ocr_fused_ctc` → read `token_ids` + `max_probs` (1 KB) | `decodeTextFromArgmax` |

```powershell
.\scripts\run_device_ctc.ps1              # default local_char=512
.\scripts\run_device_ctc.ps1 -LocalChar 512
```

## Build & run

```powershell
cd ocr_softmax_bench
node generate_testdata.js          # if testdata/logits.bin missing
.\scripts\build_ohos_ocl_test.ps1
.\scripts\run_device_ctc.ps1
```

Optional:

```powershell
.\scripts\run_device_ctc.ps1 -Runs 30 -Warmup 5
```

## Expected output

```text
decode correctness: PASS
path            kernel       read     decode      total
opt_2d+decode   ~2.XXX ms   X.XXX ms   ~7.XXX ms   X.XXX ms
fused+decode    X.XXX ms   X.XXX ms   X.XXX ms   X.XXX ms
```

## Files pushed to device

| Remote path | Local file |
|-------------|------------|
| `/data/vendor/camera/ocl_test_ctc` | `build/ohos-ocl-test/ocl_test_ctc` |
| `softmax_ocr_opt_2d.cl` | production baseline kernel |
| `softmax_ocr_fused_ctc.cl` | fused kernel |
| `testdata/logits.bin` | input |

## Notes

- Uses `dlopen` for `libOpenCL.so` on HarmonyOS (same as `ocl_test_2d`).
- CTC blank index = **0** (`device_test/ctc_decode.h`).
- `probs_ref.bin` is not required for CTC test (decode compared original vs fused on same logits).
