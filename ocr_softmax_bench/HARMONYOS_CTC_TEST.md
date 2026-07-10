# HarmonyOS CTC pipeline device test

Compare on phone:

| Path | GPU | CPU |
|------|-----|-----|
| **original** | `softmax_ocr_opt` → read `probs` (4.87 MB) | `decodeTextFromProbs` (argmax × 128) |
| **fused** | `softmax_ocr_fused_ctc` → read `token_ids` + `max_probs` (1 KB) | `decodeTextFromArgmax` (CTC collapse) |

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
decode correctness: PASS (emitted=128 tokens)
path            kernel       read     decode      total
original        X.XXX ms   X.XXX ms   X.XXX ms   X.XXX ms
fused           X.XXX ms   X.XXX ms   X.XXX ms   X.XXX ms

=== speedup ===
  e2e total:     X.XXx
```

## Files pushed to device

| Remote path | Local file |
|-------------|------------|
| `/data/vendor/camera/ocl_test_ctc` | `build/ohos-ocl-test/ocl_test_ctc` |
| `softmax_ocr_opt.cl` | kernel |
| `softmax_ocr_fused_ctc.cl` | fused kernel |
| `testdata/logits.bin` | input |

## Notes

- Uses `dlopen` for `libOpenCL.so` on HarmonyOS (same as `ocl_test_2d`).
- CTC blank index = **0** (`device_test/ctc_decode.h`).
- `probs_ref.bin` is not required for CTC test (decode compared original vs fused on same logits).
