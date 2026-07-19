#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: ocr_example_test.py /path/to/hunyuan_ocr_example", file=sys.stderr)
        return 2

    binary = Path(sys.argv[1])
    if not binary.is_file():
        print(f"OCR example binary is missing: {binary}", file=sys.stderr)
        return 1

    completed = subprocess.run(
        [str(binary)],
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        completed.returncode != 1
        or "Usage:" not in completed.stderr
        or "<model-dir> <image>" not in completed.stderr
    ):
        print(
            "OCR example did not expose the minimal two-argument contract: "
            f"rc={completed.returncode} stderr={completed.stderr!r}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
