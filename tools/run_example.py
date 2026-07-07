#!/usr/bin/env python3
"""Run one bundled HunyuanOCR-ncnn example image."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExampleCase:
    key: str
    name: str
    image: str
    prompt_mode: str
    description: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_cases(root: Path | None = None) -> list[ExampleCase]:
    if root is None:
        root = repo_root()
    manifest = root / "examples/regression_cases.json"
    items = json.loads(manifest.read_text(encoding="utf-8"))
    return [ExampleCase(**item) for item in items]


CASES = load_cases()
CASE_BY_KEY = {case.key: case for case in CASES}
CASE_BY_NAME = {case.name: case for case in CASES}


def default_binary(root: Path) -> Path:
    candidates = [
        root / "build/hunyuan_ocr_cli",
        root / "build/Release/hunyuan_ocr_cli.exe",
        root / "build/RelWithDebInfo/hunyuan_ocr_cli.exe",
        root / "build/Debug/hunyuan_ocr_cli.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def resolve_case(value: str) -> ExampleCase:
    if value in CASE_BY_KEY:
        return CASE_BY_KEY[value]
    if value in CASE_BY_NAME:
        return CASE_BY_NAME[value]
    valid = ", ".join(case.key for case in CASES)
    raise SystemExit(f"unknown case: {value}\nvalid cases: {valid}")


def print_case_list() -> None:
    for case in CASES:
        print(f"{case.key}\t{case.prompt_mode}\t{case.image}\t{case.description}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")


def build_command(
    binary: Path,
    model: Path,
    image: Path,
    prompt_mode: str,
    prompt: str | None,
    max_tokens: int | None,
) -> list[str]:
    cmd = [
        str(binary),
        "--model",
        str(model),
        "--image",
        str(image),
    ]
    if prompt is not None:
        cmd.extend(["--prompt", prompt])
    else:
        cmd.extend(["--prompt-mode", prompt_mode])
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])
    return cmd


def run_case(
    case: ExampleCase,
    *,
    binary: Path,
    model: Path,
    image_root: Path,
    prompt: str | None = None,
    max_tokens: int | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    image = image_root / case.image
    require_file(binary, "hunyuan_ocr_cli")
    require_dir(model, "model directory")
    require_file(image, f"image for {case.key}")

    cmd = build_command(binary, model, image, case.prompt_mode, prompt, max_tokens)
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=repo_root(), text=True, capture_output=capture_output)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run one bundled HunyuanOCR-ncnn example image.")
    parser.add_argument("--model", type=Path, help="Packaged HunyuanOCR-ncnn model directory.")
    parser.add_argument("--case", default="hf_demo", help="Example case key. Use --list to show keys.")
    parser.add_argument("--binary", type=Path, default=default_binary(root), help="Path to hunyuan_ocr_cli.")
    parser.add_argument("--image-root", type=Path, default=root / "examples/images", help="Example image directory.")
    parser.add_argument("--prompt", default=None, help="Override the case prompt mode with custom prompt text.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional generation token limit.")
    parser.add_argument("--list", action="store_true", help="List bundled example cases and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print_case_list()
        return 0
    if args.model is None:
        raise SystemExit("--model is required unless --list is used")

    case = resolve_case(args.case)
    completed = run_case(
        case,
        binary=args.binary.resolve(),
        model=args.model.resolve(),
        image_root=args.image_root.resolve(),
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
