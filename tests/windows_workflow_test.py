#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github/workflows/windows-compile.yml").read_text(encoding="utf-8")
    required = (
        "workflow_dispatch:",
        "runs-on: windows-2022",
        "NCNN_REF: 244f30c8b995d5b2cf57b59950596490c68813d6",
        "ref: ${{ env.NCNN_REF }}",
        '-G "Visual Studio 17 2022" -A x64',
        "-DHUNYUAN_OCR_BUILD_TESTS=ON",
        "-DHUNYUAN_OCR_BUILD_TOKENIZER_TEST=OFF",
        "cmake --build build --config Release --parallel",
        "ctest --test-dir build -C Release --output-on-failure",
        "--no-tests=error",
        "precise_sdpa|multimodal_rope|utf8",
        "msys2/setup-msys2@v2",
        "msystem: UCRT64",
        "mingw-w64-ucrt-x86_64-gcc",
        "mingw-w64-ucrt-x86_64-cmake",
        "mingw-w64-ucrt-x86_64-ninja",
        "-G Ninja",
    )
    missing = [value for value in required if value not in workflow]
    if missing:
        print(f"Windows workflow missing validation settings: {missing}", file=sys.stderr)
        return 1
    generator = '-G "Visual Studio 17 2022" -A x64'
    if workflow.count(generator) != 2:
        print("Windows workflow must configure both ncnn and HunyuanOCR-ncnn for VS2022 x64", file=sys.stderr)
        return 1
    if workflow.count("NCNN_REF: 244f30c8b995d5b2cf57b59950596490c68813d6") != 2:
        print("MSVC and UCRT64 jobs must use the same pinned ncnn revision", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
