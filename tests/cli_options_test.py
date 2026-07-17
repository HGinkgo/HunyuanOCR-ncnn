#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def require_rejected(binary: Path, args: list[str], marker: str) -> bool:
    completed = subprocess.run(
        [str(binary), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0 or marker not in completed.stderr:
        print(
            "command was not rejected: "
            f"args={args!r} rc={completed.returncode} stderr={completed.stderr!r}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: cli_options_test.py /path/to/hunyuan_ocr_cli", file=sys.stderr)
        return 2

    binary = Path(sys.argv[1])
    completed = subprocess.run(
        [
            str(binary),
            "--repetition-penalty",
            "1.08",
            "--mmap-weights",
            "--version",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        return 1
    if "HunyuanOCR-ncnn 0.4.0" not in completed.stdout:
        print("0.4.0 version output missing after --repetition-penalty", file=sys.stderr)
        return 1
    if (
        "vision Vulkan support: enabled" not in completed.stdout
        and "vision Vulkan support: disabled" not in completed.stdout
    ):
        print("version output does not report vision Vulkan support", file=sys.stderr)
        return 1
    vulkan_compiled = "vision Vulkan support: enabled" in completed.stdout

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
        or "--vision-vulkan" not in help_result.stdout
        or "--vision-vulkan-device" not in help_result.stdout
        or "--mmap-weights" not in help_result.stdout
        or "--batch-input" not in help_result.stdout
        or "--batch-output" not in help_result.stdout
        or "--force" not in help_result.stdout
        or "--vlm-fixture or --image with --prompt/--prompt-mode"
        not in help_result.stdout
    ):
        print("CLI help does not report the expected 1.5/DFlash options", file=sys.stderr)
        return 1

    if not require_rejected(
        binary,
        ["--model", ".", "--batch-input", "requests.jsonl"],
        "--batch-input and --batch-output must be provided together",
    ):
        return 1
    if not require_rejected(
        binary,
        ["--model", ".", "--batch-output", "results.jsonl"],
        "--batch-input and --batch-output must be provided together",
    ):
        return 1
    if not require_rejected(
        binary,
        [
            "--model",
            ".",
            "--batch-input",
            "requests.jsonl",
            "--batch-output",
            "results.jsonl",
            "--image",
            "x.png",
        ],
        "--batch-input and --image are mutually exclusive",
    ):
        return 1
    if not require_rejected(
        binary,
        [
            "--model",
            ".",
            "--batch-input",
            "requests.jsonl",
            "--batch-output",
            "results.jsonl",
            "--prompt-mode",
            "document",
        ],
        "batch prompts must be specified in JSONL records",
    ):
        return 1
    if not require_rejected(
        binary,
        ["--model", ".", "--force"],
        "--force requires batch mode",
    ):
        return 1
    if not require_rejected(
        binary,
        [
            "--model",
            ".",
            "--batch-input",
            "requests.jsonl",
            "--batch-output",
            "results.jsonl",
            "--benchmark",
        ],
        "batch mode does not support diagnostic or benchmark options",
    ):
        return 1
    if not require_rejected(
        binary,
        [
            "--model",
            ".",
            "--batch-input",
            "requests.jsonl",
            "--batch-output",
            "results.jsonl",
            "--vlm-fixture",
            ".",
        ],
        "batch mode does not support diagnostic or benchmark options",
    ):
        return 1

    required_model_files = (
        "tokenizer/vocab.txt",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens.json",
        "tokenizer/eos_ids.json",
        "text_embed/text_embed.ncnn.param",
        "text_embed/text_embed.ncnn.bin",
        "text_decoder/text_decoder_kv.ncnn.param",
        "text_decoder/text_decoder_kv.ncnn.bin",
        "lm_head/lm_head.ncnn.param",
        "lm_head/lm_head.ncnn.bin",
    )
    with tempfile.TemporaryDirectory(prefix="hunyuan_batch_preflight_") as temporary:
        temporary_root = Path(temporary)
        model_root = temporary_root / "model"
        for relative_path in required_model_files:
            path = model_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        preflight = subprocess.run(
            [
                str(binary),
                "--model",
                str(model_root),
                "--batch-input",
                str(temporary_root / "missing.jsonl"),
                "--batch-output",
                str(temporary_root / "results.jsonl"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            preflight.returncode == 0
            or "Batch failed at batch_input" not in preflight.stderr
            or "Runtime load failed" in preflight.stderr
        ):
            print(
                "batch input was not validated before runtime load: "
                f"rc={preflight.returncode} stderr={preflight.stderr!r}",
                file=sys.stderr,
            )
            return 1

    if not require_rejected(
        binary,
        ["--model", ".", "--vision-vulkan-device", "0"],
        "--vision-vulkan-device requires --vision-vulkan",
    ):
        return 1
    if not require_rejected(
        binary,
        ["--model", ".", "--vision-vulkan", "--vision-vulkan-device", "-1"],
        "--vision-vulkan-device must be non-negative",
    ):
        return 1
    if not require_rejected(
        binary,
        ["--model", ".", "--vision-vulkan", "--vision-vulkan-device", "gpu"],
        "--vision-vulkan-device value must be an integer",
    ):
        return 1
    if not require_rejected(
        binary,
        ["--model", ".", "--vlm-fixture", ".", "--vision-vulkan"],
        "--vision-vulkan requires a path that executes vision",
    ):
        return 1
    if not require_rejected(
        binary,
        ["--model", ".", "--image", "x.png", "--vision-vulkan"],
        "--vision-vulkan requires a path that executes vision",
    ):
        return 1
    if not require_rejected(
        binary,
        ["--model", ".", "--image-file-fixture", ".", "--vision-vulkan"],
        "--vision-vulkan requires a path that executes vision",
    ):
        return 1
    if not vulkan_compiled:
        if not require_rejected(
            binary,
            [
                "--model",
                ".",
                "--image",
                "x.png",
                "--prompt-mode",
                "document",
                "--vision-vulkan",
            ],
            "ncnn was built without Vulkan support",
        ):
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
