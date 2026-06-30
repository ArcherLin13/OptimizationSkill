#pragma once
namespace sc_ocv_dft {
void dft1dComplex32fInplace(float* interleaved, int len, bool inverse, bool scaled);
}
