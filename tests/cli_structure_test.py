#!/usr/bin/env python3
"""Keep the CLI entry point split into focused implementation units."""

from __future__ import annotations

import sys
from pathlib import Path


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    main_source = (root / "src/main.cpp").read_text(encoding="utf-8")
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")

    modules = ("cli_options", "cli_fixture", "cli_benchmark")
    module_sources: dict[str, str] = {}
    for module in modules:
        for suffix in (".h", ".cpp"):
            path = root / "src" / f"{module}{suffix}"
            if not path.is_file():
                return fail(f"missing CLI module: src/{path.name}")
        source_name = f"{module}.cpp"
        if f"src/{source_name}" not in cmake:
            return fail(f"CLI target does not compile src/{source_name}")
        module_sources[module] = (root / "src" / source_name).read_text(encoding="utf-8")

    line_count = len(main_source.splitlines())
    if line_count > 900:
        return fail(f"src/main.cpp remains too large: {line_count} lines")

    forbidden = (
        "void print_usage(",
        "std::string discover_model_root(",
        "int print_dflash_decode_result(",
        "int run_vlm_decode_with_features(",
        "struct BenchmarkTiming",
        "int run_image_benchmark(",
    )
    for signature in forbidden:
        if signature in main_source:
            return fail(f"src/main.cpp still owns extracted implementation: {signature}")

    benchmark_fields = (
        "cold_start_total_ms",
        "vision_load_ms",
        "text_load_ms",
        "tokenizer_load_ms",
        "warm_inference_total_ms",
        "generated_token_per_s",
        "decode_token_per_s",
    )
    for field in benchmark_fields:
        if f'"  {field}: "' not in module_sources["cli_benchmark"]:
            return fail(f"benchmark output field missing after extraction: {field}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
