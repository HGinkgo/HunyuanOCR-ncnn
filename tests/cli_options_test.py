#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: cli_options_test.py /path/to/hunyuan_ocr_cli", file=sys.stderr)
        return 2

    binary = Path(sys.argv[1])
    completed = subprocess.run(
        [str(binary), "--repetition-penalty", "1.08", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        return 1
    if "HunyuanOCR-ncnn" not in completed.stdout:
        print("version output missing after --repetition-penalty", file=sys.stderr)
        return 1

    help_result = subprocess.run(
        [str(binary), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    if help_result.returncode != 0 or "Default: 1.08." not in help_result.stdout:
        print("CLI help does not report the HunyuanOCR 1.5 repetition penalty default", file=sys.stderr)
        return 1

    invalid = subprocess.run(
        [str(binary), "--repetition-penalty", "0", "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if invalid.returncode == 0 or "must be positive" not in invalid.stderr:
        print("non-positive repetition penalty was not rejected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
