#!/usr/bin/env python3
"""Benchmark bundled HunyuanOCR-ncnn example cases."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from run_example import CASES, ExampleCase, default_binary, repo_root, require_dir, require_file


BENCHMARK_KEYS = [
    "preprocess_ms",
    "vision_ms",
    "prompt_ms",
    "text_load_ms",
    "text_embed_ms",
    "prefill_ms",
    "decode_ms",
    "text_total_ms",
    "total_ms",
    "generated_token_count",
    "generated_token_per_s",
    "decode_token_per_s",
]


@dataclass(frozen=True)
class BenchmarkRow:
    case: str
    prompt_mode: str
    image: str
    repeat: int
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run simple HunyuanOCR-ncnn runtime benchmarks.")
    parser.add_argument("--model", type=Path, required=True, help="Packaged HunyuanOCR-ncnn model directory.")
    parser.add_argument("--binary", type=Path, default=default_binary(root), help="Path to hunyuan_ocr_cli.")
    parser.add_argument("--image-root", type=Path, default=root / "examples/images", help="Example image directory.")
    parser.add_argument(
        "--cases",
        default="hf_demo",
        help="Comma-separated case keys, or 'all'. Use tools/run_example.py --list to see keys.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Measured repeats per case. Default: 1.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs per case. Default: 0.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Generation token limit. Default: 64.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def resolve_cases(value: str) -> list[ExampleCase]:
    if value == "all":
        return list(CASES)
    by_key = {case.key: case for case in CASES}
    selected: list[ExampleCase] = []
    for key in [item.strip() for item in value.split(",") if item.strip()]:
        if key not in by_key:
            valid = ", ".join(case.key for case in CASES)
            raise SystemExit(f"unknown case: {key}\nvalid cases: {valid}")
        selected.append(by_key[key])
    if not selected:
        raise SystemExit("--cases must not be empty")
    return selected


def parse_benchmark(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    in_block = False
    for line in stdout.splitlines():
        if line == "Benchmark:":
            in_block = True
            continue
        if not in_block:
            continue
        if not line.startswith("  ") or ": " not in line:
            break
        key, value = line.strip().split(": ", 1)
        if key in BENCHMARK_KEYS:
            metrics[key] = float(value)
    missing = [key for key in BENCHMARK_KEYS if key not in metrics]
    if missing:
        raise ValueError("missing benchmark fields: " + ", ".join(missing))
    return metrics


def run_once(
    *,
    binary: Path,
    model: Path,
    image: Path,
    prompt_mode: str,
    max_tokens: int,
) -> dict[str, float]:
    cmd = [
        str(binary),
        "--model",
        str(model),
        "--image",
        str(image),
        "--prompt-mode",
        prompt_mode,
        "--max-tokens",
        str(max_tokens),
        "--benchmark",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(f"benchmark command failed with exit code {completed.returncode}: {' '.join(cmd)}")
    try:
        return parse_benchmark(completed.stdout)
    except ValueError as exc:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(f"failed to parse benchmark output: {exc}") from exc


def average_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(sample[key] for sample in samples) / float(len(samples))
        for key in BENCHMARK_KEYS
    }


def print_table(rows: list[BenchmarkRow]) -> None:
    header = [
        "case",
        "mode",
        "repeat",
        "tokens",
        "preprocess",
        "vision",
        "prefill",
        "decode",
        "total",
        "decode tok/s",
    ]
    print("\t".join(header))
    for row in rows:
        metrics = row.metrics
        values = [
            row.case,
            row.prompt_mode,
            str(row.repeat),
            f"{metrics['generated_token_count']:.0f}",
            f"{metrics['preprocess_ms']:.1f}",
            f"{metrics['vision_ms']:.1f}",
            f"{metrics['prefill_ms']:.1f}",
            f"{metrics['decode_ms']:.1f}",
            f"{metrics['total_ms']:.1f}",
            f"{metrics['decode_token_per_s']:.2f}",
        ]
        print("\t".join(values))


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case", "prompt_mode", "image", "repeat", *BENCHMARK_KEYS],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case": row.case,
                    "prompt_mode": row.prompt_mode,
                    "image": row.image,
                    "repeat": row.repeat,
                    **{key: f"{row.metrics[key]:.6f}" for key in BENCHMARK_KEYS},
                }
            )


def main() -> int:
    args = parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    binary = args.binary.resolve()
    model = args.model.resolve()
    image_root = args.image_root.resolve()
    require_file(binary, "hunyuan_ocr_cli")
    require_dir(model, "model directory")
    require_dir(image_root, "image root")

    rows: list[BenchmarkRow] = []
    for case in resolve_cases(args.cases):
        image = image_root / case.image
        require_file(image, f"image for {case.key}")
        for _ in range(args.warmup):
            run_once(
                binary=binary,
                model=model,
                image=image,
                prompt_mode=case.prompt_mode,
                max_tokens=args.max_tokens,
            )
        samples = [
            run_once(
                binary=binary,
                model=model,
                image=image,
                prompt_mode=case.prompt_mode,
                max_tokens=args.max_tokens,
            )
            for _ in range(args.repeat)
        ]
        rows.append(
            BenchmarkRow(
                case=case.key,
                prompt_mode=case.prompt_mode,
                image=case.image,
                repeat=args.repeat,
                metrics=average_metrics(samples),
            )
        )

    print_table(rows)
    if args.csv is not None:
        write_csv(args.csv.resolve(), rows)
        print(f"csv: {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
