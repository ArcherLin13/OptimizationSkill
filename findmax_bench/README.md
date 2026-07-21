# findMaxValue OpenCL bench

Compare three kernels on `half` image (`5760×4320` default):

| impl | idea |
|---|---|
| `orig_2d_1px` | original 2D, 1 px/WI, local `float[256]` reduce, atomic/WG |
| `base_1d_1px` | same algorithm, 1D launch |
| `opt_stride` | **OCR-softmax style**: grid-stride multi-px/WI + `half4` + local reduce; only `nwg` atomics |

`opt` launch: `gws = lws_opt * nwg` (default 256×256), **not** `width*height`.

## Build / run

```powershell
cd findmax_bench
.\scripts\build_ohos.ps1
.\scripts\run_device.ps1
# tune occupancy if needed:
.\scripts\run_device.ps1 -Nwg 128
.\scripts\run_device.ps1 -Nwg 512 -LwsOpt 256
```
