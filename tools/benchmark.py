#!/usr/bin/env python3
"""Profile bundled HunyuanOCR-ncnn cases with reusable in-process runtimes."""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from run_example import CASES, ExampleCase, default_binary, repo_root, require_dir, require_file


BENCHMARK_KEYS = [
    "num_threads",
    "warmup",
    "repeat",
    "mmap_weights",
    "mapped_weight_bytes",
    "cold_start_total_ms",
    "vision_load_ms",
    "text_load_ms",
    "tokenizer_load_ms",
    "preprocess_ms",
    "vision_infer_ms",
    "prompt_ms",
    "text_embed_ms",
    "prefill_ms",
    "decode_ms",
    "lm_head_ms",
    "token_select_ms",
    "tokenizer_decode_ms",
    "text_total_ms",
    "warm_inference_total_ms",
    "generated_token_count",
    "generated_token_per_s",
    "decode_token_per_s",
]


@dataclass(frozen=True)
class BenchmarkRow:
    case: str
    prompt_mode: str
    image: str
    threads: int
    metrics: dict[str, float]
    generated_tokens: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Profile HunyuanOCR-ncnn runtime stages.")
    parser.add_argument("--model", type=Path, required=True, help="Packaged HunyuanOCR-ncnn model directory.")
    parser.add_argument("--binary", type=Path, default=default_binary(root), help="Path to hunyuan_ocr_cli.")
    parser.add_argument("--image-root", type=Path, default=root / "examples/images", help="Example image directory.")
    parser.add_argument(
        "--cases",
        default="hf_demo",
        help="Comma-separated case keys, or 'all'. Use tools/run_example.py --list to see keys.",
    )
    parser.add_argument(
        "--threads",
        default="0",
        help="Comma-separated ncnn thread counts. Use 0 for the ncnn default. Default: 0.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Same-process measured repeats per case. Default: 1.")
    parser.add_argument("--warmup", type=int, default=0, help="Same-process warmup runs per case. Default: 0.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Generation token limit. Default: 64.")
    parser.add_argument(
        "--mmap-weights",
        action="store_true",
        help="Load model weights from read-only file mappings.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional combined CSV output path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for environment.txt, cold_start.csv, warm_inference.csv, and summary.md.",
    )
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


def resolve_threads(value: str) -> list[int]:
    threads: list[int] = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        try:
            count = int(item)
        except ValueError as exc:
            raise SystemExit(f"invalid thread count: {item}") from exc
        if count < 0:
            raise SystemExit("thread counts must be non-negative")
        if count not in threads:
            threads.append(count)
    if not threads:
        raise SystemExit("--threads must not be empty")
    return threads


def parse_benchmark(stdout: str) -> tuple[dict[str, float], tuple[int, ...]]:
    metrics: dict[str, float] = {}
    generated_tokens: tuple[int, ...] | None = None
    in_block = False
    for line in stdout.splitlines():
        if line == "Benchmark:":
            in_block = True
            continue
        if line.startswith("Benchmark generated tokens:"):
            text = line.split(":", 1)[1].strip()
            generated_tokens = tuple(int(item) for item in text.split()) if text else ()
            continue
        if not in_block:
            continue
        if not line.startswith("  ") or ": " not in line:
            in_block = False
            continue
        key, value = line.strip().split(": ", 1)
        if key in BENCHMARK_KEYS:
            metrics[key] = float(value)
    missing = [key for key in BENCHMARK_KEYS if key not in metrics]
    if missing:
        raise ValueError("missing benchmark fields: " + ", ".join(missing))
    if generated_tokens is None:
        raise ValueError("missing benchmark generated token ids")
    return metrics, generated_tokens


def run_benchmark(
    *,
    binary: Path,
    model: Path,
    image: Path,
    prompt_mode: str,
    max_tokens: int,
    threads: int,
    warmup: int,
    repeat: int,
    mmap_weights: bool,
) -> tuple[dict[str, float], tuple[int, ...]]:
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
        "--benchmark-warmup",
        str(warmup),
        "--benchmark-repeat",
        str(repeat),
    ]
    if threads > 0:
        cmd.extend(["--num-threads", str(threads)])
    if mmap_weights:
        cmd.append("--mmap-weights")
    process_start = time.perf_counter()
    completed = subprocess.run(cmd, text=True, capture_output=True)
    process_wall_ms = (time.perf_counter() - process_start) * 1000.0
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(f"benchmark command failed with exit code {completed.returncode}: {' '.join(cmd)}")
    try:
        metrics, generated_tokens = parse_benchmark(completed.stdout)
    except ValueError as exc:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(f"failed to parse benchmark output: {exc}") from exc
    metrics["benchmark_process_wall_ms"] = process_wall_ms
    return metrics, generated_tokens


def print_table(rows: list[BenchmarkRow]) -> None:
    header = [
        "case",
        "threads",
        "tokens",
        "cold",
        "warm",
        "vision load",
        "vision infer",
        "prefill",
        "decoder",
        "lm_head",
        "decode tok/s",
    ]
    print("\t".join(header))
    for row in rows:
        metrics = row.metrics
        values = [
            row.case,
            str(row.threads),
            f"{metrics['generated_token_count']:.0f}",
            f"{metrics['cold_start_total_ms']:.1f}",
            f"{metrics['warm_inference_total_ms']:.1f}",
            f"{metrics['vision_load_ms']:.1f}",
            f"{metrics['vision_infer_ms']:.1f}",
            f"{metrics['prefill_ms']:.1f}",
            f"{metrics['decode_ms']:.1f}",
            f"{metrics['lm_head_ms']:.1f}",
            f"{metrics['decode_token_per_s']:.2f}",
        ]
        print("\t".join(values))


def combined_fields() -> list[str]:
    return [
        "case",
        "prompt_mode",
        "image",
        "threads",
        "generated_tokens",
        "benchmark_process_wall_ms",
        *BENCHMARK_KEYS,
    ]


def row_dict(row: BenchmarkRow) -> dict[str, str | int]:
    return {
        "case": row.case,
        "prompt_mode": row.prompt_mode,
        "image": row.image,
        "threads": row.threads,
        "generated_tokens": " ".join(str(token) for token in row.generated_tokens),
        "benchmark_process_wall_ms": f"{row.metrics['benchmark_process_wall_ms']:.6f}",
        **{key: f"{row.metrics[key]:.6f}" for key in BENCHMARK_KEYS},
    }


def write_csv(path: Path, rows: list[BenchmarkRow], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_dict(row))


def write_output_dir(path: Path, rows: list[BenchmarkRow], command: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cold_fields = [
        "case",
        "prompt_mode",
        "image",
        "threads",
        "benchmark_process_wall_ms",
        "cold_start_total_ms",
        "vision_load_ms",
        "text_load_ms",
        "tokenizer_load_ms",
        "generated_tokens",
    ]
    warm_fields = [
        "case",
        "prompt_mode",
        "image",
        "threads",
        "warmup",
        "repeat",
        "generated_token_count",
        "preprocess_ms",
        "vision_infer_ms",
        "prompt_ms",
        "text_embed_ms",
        "prefill_ms",
        "decode_ms",
        "lm_head_ms",
        "token_select_ms",
        "tokenizer_decode_ms",
        "text_total_ms",
        "warm_inference_total_ms",
        "generated_token_per_s",
        "decode_token_per_s",
        "generated_tokens",
    ]
    write_csv(path / "cold_start.csv", rows, cold_fields)
    write_csv(path / "warm_inference.csv", rows, warm_fields)

    environment = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"python={platform.python_version()}",
        f"command={command}",
    ]
    (path / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")

    summary = [
        "# P5.1 Benchmark Summary",
        "",
        "| case | threads | tokens | cold ms | warm ms | vision infer ms | prefill ms | decoder ms | lm_head ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row.metrics
        summary.append(
            f"| {row.case} | {row.threads} | {metrics['generated_token_count']:.0f} "
            f"| {metrics['cold_start_total_ms']:.1f} | {metrics['warm_inference_total_ms']:.1f} "
            f"| {metrics['vision_infer_ms']:.1f} | {metrics['prefill_ms']:.1f} "
            f"| {metrics['decode_ms']:.1f} | {metrics['lm_head_ms']:.1f} |"
        )
    (path / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


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
    reference_tokens: dict[str, tuple[int, ...]] = {}
    for case in resolve_cases(args.cases):
        image = image_root / case.image
        require_file(image, f"image for {case.key}")
        for threads in resolve_threads(args.threads):
            label = str(threads) if threads > 0 else "default"
            print(f"running case={case.key} threads={label}", flush=True)
            metrics, generated_tokens = run_benchmark(
                binary=binary,
                model=model,
                image=image,
                prompt_mode=case.prompt_mode,
                max_tokens=args.max_tokens,
                threads=threads,
                warmup=args.warmup,
                repeat=args.repeat,
                mmap_weights=args.mmap_weights,
            )
            expected = reference_tokens.setdefault(case.key, generated_tokens)
            if generated_tokens != expected:
                raise SystemExit(
                    f"generated token mismatch across thread settings for {case.key}: "
                    f"reference={expected} threads={threads} actual={generated_tokens}"
                )
            rows.append(
                BenchmarkRow(
                    case=case.key,
                    prompt_mode=case.prompt_mode,
                    image=case.image,
                    threads=int(metrics["num_threads"]),
                    metrics=metrics,
                    generated_tokens=generated_tokens,
                )
            )

    print_table(rows)
    if args.csv is not None:
        csv_path = args.csv.resolve()
        write_csv(csv_path, rows, combined_fields())
        print(f"csv: {csv_path}")
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        write_output_dir(output_dir, rows, " ".join(sys.argv))
        print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
