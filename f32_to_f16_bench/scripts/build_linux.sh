#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build/linux"
HEADERS="${ROOT}/../ocr_softmax_bench/third_party/OpenCL-Headers"
if [[ ! -f "${HEADERS}/CL/cl.h" ]]; then
  echo "Missing OpenCL headers: ${HEADERS}/CL/cl.h" >&2
  exit 1
fi
mkdir -p "${BUILD}"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release -DOPENCL_HEADERS_DIR="${HEADERS}"
cmake --build "${BUILD}" --target ocl_test_f32_to_f16
echo "Built: ${BUILD}/ocl_test_f32_to_f16"
echo "Run from ${ROOT}: ./build/linux/ocl_test_f32_to_f16"
