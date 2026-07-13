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
    if "HunyuanOCR-ncnn 0.3.0" not in completed.stdout:
        print("0.3.0 version output missing after --repetition-penalty", file=sys.stderr)
        return 1

    help_result = subprocess.run(
        [str(binary), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        help_result.returncode != 0
        or "Default: 1.08." not in help_result.stdout
        or "--dflash-probe" not in help_result.stdout
        or "--dflash             " not in help_result.stdout
        or "--vlm-fixture or --image with --prompt/--prompt-mode"
        not in help_result.stdout
    ):
        print("CLI help does not report the expected 1.5/DFlash options", file=sys.stderr)
        return 1

    dflash_without_fixture = subprocess.run(
        [str(binary), "--model", ".", "--dflash"],
        text=True,
        capture_output=True,
        check=False,
    )
    dflash_need_msg = (
        "--vlm-fixture or --image with --prompt/--prompt-mode"
    )
    if (
        dflash_without_fixture.returncode == 0
        or dflash_need_msg not in dflash_without_fixture.stderr
    ):
        print(
            "--dflash without --vlm-fixture or --image/--prompt was not rejected: "
            f"rc={dflash_without_fixture.returncode} stderr={dflash_without_fixture.stderr!r}",
            file=sys.stderr,
        )
        return 1

    dflash_benchmark = subprocess.run(
        [
            str(binary),
            "--model",
            ".",
            "--image",
            "x.png",
            "--prompt-mode",
            "spotting",
            "--benchmark",
            "--dflash",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        dflash_benchmark.returncode == 0
        or "--benchmark does not support --dflash yet" not in dflash_benchmark.stderr
    ):
        print(
            "--benchmark --dflash was not rejected: "
            f"rc={dflash_benchmark.returncode} stderr={dflash_benchmark.stderr!r}",
            file=sys.stderr,
        )
        return 1

    dflash_probe_mutex = subprocess.run(
        [
            str(binary),
            "--model",
            ".",
            "--vlm-fixture",
            ".",
            "--dflash",
            "--dflash-probe",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        dflash_probe_mutex.returncode == 0
        or "--dflash and --dflash-probe are mutually exclusive"
        not in dflash_probe_mutex.stderr
    ):
        print(
            "--dflash/--dflash-probe mutual exclusion failed: "
            f"rc={dflash_probe_mutex.returncode} stderr={dflash_probe_mutex.stderr!r}",
            file=sys.stderr,
        )
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
