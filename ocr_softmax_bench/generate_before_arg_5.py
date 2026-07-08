#!/usr/bin/env python3
"""Generate before_arg_5.bin for softmax_ocr_opt_2d local-memory arg (reduce_buf).

reduce_buf is OpenCL __local scratch (kernel arg index 4 = the 5th argument).
At kernel entry its contents are undefined; many test harnesses still want a
placeholder binary of the correct byte size. This script writes zeros.

Size: LOCAL_CHAR * sizeof(float)
  LOCAL_CHAR=512 -> 2048 bytes  (recommended launch)
  LOCAL_CHAR=256 -> 1024 bytes

Usage:
  python generate_before_arg_5.py
  python generate_before_arg_5.py --local-char 256 -o my_before_arg_5.bin
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

DEFAULT_LOCAL_CHAR = 512
OUTPUT_NAME = "before_arg_5.bin"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate reduce_buf placeholder binary (arg 5)")
    p.add_argument(
        "--local-char",
        type=int,
        default=DEFAULT_LOCAL_CHAR,
        help=f"LOCAL_CHAR / local_size[1] (default: {DEFAULT_LOCAL_CHAR})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "testdata" / OUTPUT_NAME,
        help=f"output path (default: testdata/{OUTPUT_NAME})",
    )
    p.add_argument(
        "--fill",
        choices=("zero", "nan"),
        default="zero",
        help="fill pattern (zero is correct for pre-kernel scratch)",
    )
    args = p.parse_args()

    if args.local_char <= 0 or (args.local_char & (args.local_char - 1)) != 0:
        raise SystemExit("local-char must be a positive power of two (128/256/512)")

    n_floats = args.local_char
    n_bytes = n_floats * 4

    if args.fill == "zero":
        payload = b"\x00" * n_bytes
    else:
        payload = struct.pack(f"<{n_floats}f", *([float("nan")] * n_floats))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    print(f"Wrote {args.output}")
    print(f"  kernel arg: index 4 (5th arg) = reduce_buf / __local float[{args.local_char}]")
    print(f"  bytes: {n_bytes}  (= LOCAL_CHAR {args.local_char} x 4)")
    print(f"  fill: {args.fill}")
    print()
    print("Note: real OpenCL launch uses clSetKernelArg(k, 4, size, nullptr);")
    print("      GPU allocates local memory — this file is only for your test platform.")


if __name__ == "__main__":
    main()
