#!/usr/bin/env python3
"""Profile bundled HunyuanOCR-ncnn cases with reusable in-process runtimes."""

from __future__ import annotations

import argparse
import csv
import platform
import shutil
import subprocess
import sys
import threading
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

PROCESS_METRIC_KEYS = [
    "benchmark_process_wall_ms",
    "benchmark_process_memory_supported",
    "benchmark_process_peak_rss_bytes",
    "benchmark_process_max_sampled_rss_anon_bytes",
    "benchmark_process_max_sampled_rss_file_bytes",
    "benchmark_device_memory_supported",
    "benchmark_device_memory_device",
    "benchmark_device_memory_baseline_bytes",
    "benchmark_device_memory_peak_bytes",
    "benchmark_device_memory_peak_delta_bytes",
]


@dataclass
class ProcessMemoryPeak:
    supported: bool = False
    peak_rss_bytes: int = 0
    max_sampled_rss_anon_bytes: int = 0
    max_sampled_rss_file_bytes: int = 0

    def update_from_linux_status(self, status: str) -> None:
        values: dict[str, int] = {}
        for line in status.splitlines():
            parts = line.split()
            if len(parts) != 3 or parts[2] != "kB":
                continue
            key = parts[0].removesuffix(":")
            if key not in {"VmHWM", "RssAnon", "RssFile"}:
                continue
            try:
                values[key] = int(parts[1]) * 1024
            except ValueError:
                continue
        if not {"VmHWM", "RssAnon", "RssFile"}.issubset(values):
            return
        self.supported = True
        self.peak_rss_bytes = max(self.peak_rss_bytes, values["VmHWM"])
        self.max_sampled_rss_anon_bytes = max(
            self.max_sampled_rss_anon_bytes, values["RssAnon"]
        )
        self.max_sampled_rss_file_bytes = max(
            self.max_sampled_rss_file_bytes, values["RssFile"]
        )

    def sample_linux_process(self, pid: int) -> None:
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeError):
            return
        self.update_from_linux_status(status)

    def as_metrics(self) -> dict[str, float]:
        return {
            "benchmark_process_memory_supported": float(self.supported),
            "benchmark_process_peak_rss_bytes": float(self.peak_rss_bytes),
            "benchmark_process_max_sampled_rss_anon_bytes": float(
                self.max_sampled_rss_anon_bytes
            ),
            "benchmark_process_max_sampled_rss_file_bytes": float(
                self.max_sampled_rss_file_bytes
            ),
        }


@dataclass
class DeviceMemoryPeak:
    device: int
    supported: bool = False
    baseline_bytes: int = 0
    peak_bytes: int = 0

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)

    def update_from_nvidia_smi(self, output: str) -> None:
        try:
            used_bytes = int(output.strip()) * 1024 * 1024
        except ValueError:
            return
        if not self.supported:
            self.supported = True
            self.baseline_bytes = used_bytes
        self.peak_bytes = max(self.peak_bytes, used_bytes)

    def sample_nvidia_smi(self) -> None:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--id",
                    str(self.device),
                ],
                text=True,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return
        if completed.returncode == 0:
            self.update_from_nvidia_smi(completed.stdout)

    def as_metrics(self) -> dict[str, float]:
        return {
            "benchmark_device_memory_supported": float(self.supported),
            "benchmark_device_memory_device": float(self.device),
            "benchmark_device_memory_baseline_bytes": float(self.baseline_bytes),
            "benchmark_device_memory_peak_bytes": float(self.peak_bytes),
            "benchmark_device_memory_peak_delta_bytes": float(self.peak_delta_bytes),
        }


@dataclass(frozen=True)
class BenchmarkRow:
    config: str
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
        "--config",
        default=None,
        help="Configuration label written to CSV and Markdown output.",
    )
    parser.add_argument(
        "--mmap-weights",
        action="store_true",
        help="Load model weights from read-only file mappings.",
    )
    parser.add_argument(
        "--vision-vulkan",
        action="store_true",
        help="Run the vision network with ncnn Vulkan fp32.",
    )
    parser.add_argument(
        "--vision-vulkan-device",
        type=int,
        default=0,
        help="Vulkan device index for vision. Default: 0.",
    )
    parser.add_argument(
        "--text-vulkan",
        action="store_true",
        help="Run the text decoder and lm_head with ncnn Vulkan fp32.",
    )
    parser.add_argument(
        "--text-vulkan-device",
        type=int,
        default=0,
        help="Vulkan device index for text. Default: 0.",
    )
    parser.add_argument(
        "--nvidia-smi-device",
        type=int,
        default=None,
        help=(
            "Optionally sample total used memory on one NVIDIA device once per second. "
            "The reported delta requires an otherwise idle device."
        ),
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


def build_benchmark_command(
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
    vision_vulkan: bool,
    vision_vulkan_device: int,
    text_vulkan: bool,
    text_vulkan_device: int,
) -> list[str]:
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
    if vision_vulkan:
        cmd.extend(["--vision-vulkan", "--vision-vulkan-device", str(vision_vulkan_device)])
    if text_vulkan:
        cmd.extend(["--text-vulkan", "--text-vulkan-device", str(text_vulkan_device)])
    return cmd


def default_config_label(vision_vulkan: bool, text_vulkan: bool) -> str:
    if vision_vulkan and text_vulkan:
        return "vision_text_vulkan"
    if vision_vulkan:
        return "vision_vulkan"
    if text_vulkan:
        return "text_vulkan"
    return "cpu_fp32"


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
    vision_vulkan: bool,
    vision_vulkan_device: int,
    text_vulkan: bool,
    text_vulkan_device: int,
    nvidia_smi_device: int | None,
) -> tuple[dict[str, float], tuple[int, ...]]:
    cmd = build_benchmark_command(
        binary=binary,
        model=model,
        image=image,
        prompt_mode=prompt_mode,
        max_tokens=max_tokens,
        threads=threads,
        warmup=warmup,
        repeat=repeat,
        mmap_weights=mmap_weights,
        vision_vulkan=vision_vulkan,
        vision_vulkan_device=vision_vulkan_device,
        text_vulkan=text_vulkan,
        text_vulkan_device=text_vulkan_device,
    )
    device_memory_peak = DeviceMemoryPeak(
        device=nvidia_smi_device if nvidia_smi_device is not None else -1
    )
    if nvidia_smi_device is not None:
        device_memory_peak.sample_nvidia_smi()

    process_start = time.perf_counter()
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    memory_peak = ProcessMemoryPeak()
    stop_sampling = threading.Event()

    def sample_memory() -> None:
        next_device_sample = time.monotonic()
        while not stop_sampling.wait(0.01):
            memory_peak.sample_linux_process(process.pid)
            if nvidia_smi_device is not None and time.monotonic() >= next_device_sample:
                device_memory_peak.sample_nvidia_smi()
                next_device_sample = time.monotonic() + 1.0

    sampler = threading.Thread(target=sample_memory, name="benchmark-memory-sampler")
    sampler.start()
    stdout, stderr = process.communicate()
    process_wall_ms = (time.perf_counter() - process_start) * 1000.0
    stop_sampling.set()
    sampler.join()
    if process.returncode != 0:
        print(stdout, end="")
        print(stderr, end="", file=sys.stderr)
        raise SystemExit(f"benchmark command failed with exit code {process.returncode}: {' '.join(cmd)}")
    try:
        metrics, generated_tokens = parse_benchmark(stdout)
    except ValueError as exc:
        print(stdout, end="")
        print(stderr, end="", file=sys.stderr)
        raise SystemExit(f"failed to parse benchmark output: {exc}") from exc
    metrics["benchmark_process_wall_ms"] = process_wall_ms
    metrics.update(memory_peak.as_metrics())
    metrics.update(device_memory_peak.as_metrics())
    return metrics, generated_tokens


def print_table(rows: list[BenchmarkRow]) -> None:
    header = [
        "config",
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
            row.config,
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
        "config",
        "case",
        "prompt_mode",
        "image",
        "threads",
        "generated_tokens",
        "benchmark_process_wall_ms",
        "benchmark_process_memory_supported",
        "benchmark_process_peak_rss_bytes",
        "benchmark_process_max_sampled_rss_anon_bytes",
        "benchmark_process_max_sampled_rss_file_bytes",
        "benchmark_device_memory_supported",
        "benchmark_device_memory_device",
        "benchmark_device_memory_baseline_bytes",
        "benchmark_device_memory_peak_bytes",
        "benchmark_device_memory_peak_delta_bytes",
        *BENCHMARK_KEYS,
    ]


def row_dict(row: BenchmarkRow) -> dict[str, str | int]:
    return {
        "config": row.config,
        "case": row.case,
        "prompt_mode": row.prompt_mode,
        "image": row.image,
        "threads": row.threads,
        "generated_tokens": " ".join(str(token) for token in row.generated_tokens),
        "benchmark_process_wall_ms": f"{row.metrics['benchmark_process_wall_ms']:.6f}",
        **{key: f"{row.metrics[key]:.6f}" for key in PROCESS_METRIC_KEYS},
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
        "config",
        "case",
        "prompt_mode",
        "image",
        "threads",
        "benchmark_process_wall_ms",
        "benchmark_process_memory_supported",
        "benchmark_process_peak_rss_bytes",
        "benchmark_process_max_sampled_rss_anon_bytes",
        "benchmark_process_max_sampled_rss_file_bytes",
        "benchmark_device_memory_supported",
        "benchmark_device_memory_device",
        "benchmark_device_memory_baseline_bytes",
        "benchmark_device_memory_peak_bytes",
        "benchmark_device_memory_peak_delta_bytes",
        "cold_start_total_ms",
        "vision_load_ms",
        "text_load_ms",
        "tokenizer_load_ms",
        "generated_tokens",
    ]
    warm_fields = [
        "config",
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
        "# Benchmark Summary",
        "",
        "| config | case | threads | tokens | cold ms | warm ms | vision infer ms | prefill ms | decoder ms | lm_head ms | peak RSS MiB | peak VRAM delta MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row.metrics
        vram_delta = (
            f"{metrics['benchmark_device_memory_peak_delta_bytes'] / (1024 * 1024):.1f}"
            if metrics["benchmark_device_memory_supported"]
            else "n/a"
        )
        summary.append(
            f"| {row.config} | {row.case} | {row.threads} | {metrics['generated_token_count']:.0f} "
            f"| {metrics['cold_start_total_ms']:.1f} | {metrics['warm_inference_total_ms']:.1f} "
            f"| {metrics['vision_infer_ms']:.1f} | {metrics['prefill_ms']:.1f} "
            f"| {metrics['decode_ms']:.1f} | {metrics['lm_head_ms']:.1f} "
            f"| {metrics['benchmark_process_peak_rss_bytes'] / (1024 * 1024):.1f} "
            f"| {vram_delta} |"
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
    if args.vision_vulkan_device < 0:
        raise SystemExit("--vision-vulkan-device must be non-negative")
    if args.text_vulkan_device < 0:
        raise SystemExit("--text-vulkan-device must be non-negative")
    if args.nvidia_smi_device is not None and args.nvidia_smi_device < 0:
        raise SystemExit("--nvidia-smi-device must be non-negative")

    binary = args.binary.resolve()
    model = args.model.resolve()
    image_root = args.image_root.resolve()
    require_file(binary, "hunyuan_ocr_cli")
    require_dir(model, "model directory")
    require_dir(image_root, "image root")

    rows: list[BenchmarkRow] = []
    config = args.config or default_config_label(args.vision_vulkan, args.text_vulkan)
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
                vision_vulkan=args.vision_vulkan,
                vision_vulkan_device=args.vision_vulkan_device,
                text_vulkan=args.text_vulkan,
                text_vulkan_device=args.text_vulkan_device,
                nvidia_smi_device=args.nvidia_smi_device,
            )
            expected = reference_tokens.setdefault(case.key, generated_tokens)
            if generated_tokens != expected:
                raise SystemExit(
                    f"generated token mismatch across thread settings for {case.key}: "
                    f"reference={expected} threads={threads} actual={generated_tokens}"
                )
            rows.append(
                BenchmarkRow(
                    config=config,
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
