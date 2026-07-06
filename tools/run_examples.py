#!/usr/bin/env python3
"""Run all bundled HunyuanOCR-ncnn example images."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from run_example import CASES, default_binary, repo_root, require_dir, require_file, run_case


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run all bundled HunyuanOCR-ncnn example images.")
    parser.add_argument("--model", type=Path, required=True, help="Packaged HunyuanOCR-ncnn model directory.")
    parser.add_argument("--binary", type=Path, default=default_binary(root), help="Path to hunyuan_ocr_cli.")
    parser.add_argument("--image-root", type=Path, default=root / "examples/images", help="Example image directory.")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/examples", help="Directory for logs.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional generation token limit.")
    return parser.parse_args()


def write_log(path: Path, command_line: str, stdout: str, stderr: str) -> None:
    path.write_text(
        "$ " + command_line + "\n\n"
        + "## stdout\n"
        + stdout
        + "\n## stderr\n"
        + stderr,
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    model = args.model.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()

    require_file(binary, "hunyuan_ocr_cli")
    require_dir(model, "model directory")
    require_dir(image_root, "image root")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    passed = 0
    for case in CASES:
        completed = run_case(
            case,
            binary=binary,
            model=model,
            image_root=image_root,
            max_tokens=args.max_tokens,
            capture_output=True,
        )
        log_path = output_dir / f"{case.key}.log"
        write_log(log_path, " ".join(completed.args), completed.stdout, completed.stderr)
        if completed.returncode == 0:
            passed += 1
            print(f"PASS {case.key} ({case.prompt_mode})")
        else:
            print(f"FAIL {case.key} ({case.prompt_mode}) exit={completed.returncode}", file=sys.stderr)
        print(f"  log: {log_path}")

    print(f"summary: {passed}/{len(CASES)} passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
