#pragma once
namespace sc_ocv_dft {
// OpenCV-compatible in-place 1D complex DFT (float interleaved re,im,...).
// useSimd enables ARM NEON radix-4 when SC_OCV_DFT_NEON is defined.
void dft1dComplex32fInplace(float* interleaved, int len, bool inverse, bool scaled, bool useSimd = true);
}
