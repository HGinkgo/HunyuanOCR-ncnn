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
        "python scripts/apply_ncnn_patches.py --ncnn-dir _deps/ncnn",
        '-G "Visual Studio 17 2022" -A x64',
        "-DHUNYUAN_OCR_BUILD_TESTS=ON",
        "-DHUNYUAN_OCR_BUILD_TOKENIZER_TEST=OFF",
        "cmake --build build --config Release --parallel",
        "ctest --test-dir build -C Release --output-on-failure",
        "--no-tests=error",
        "hunyuan_ocr_api|batch_jsonl|precise_sdpa|multimodal_rope|utf8|dflash_runtime|decoder_microbench_parser|text_decoder_aux|cli_options|package_model|memory_lifecycle|kv_cache_lifecycle",
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
    if workflow.count("python scripts/apply_ncnn_patches.py --ncnn-dir _deps/ncnn") != 2:
        print("MSVC and UCRT64 jobs must validate the same ncnn patch series", file=sys.stderr)
        return 1
    msvc = workflow[: workflow.index("\n  ucrt64:")]
    ucrt64 = workflow[workflow.index("\n  ucrt64:") :]
    for job_name, job in (("MSVC", msvc), ("UCRT64", ucrt64)):
        if job.count("actions/setup-python@v5") != 1 or job.count('python-version: "3.12"') != 1:
            print(f"{job_name} must explicitly set up Python 3.12 for ncnn patching", file=sys.stderr)
            return 1
        if job.index("- name: Set up Python") > job.index("- name: Apply pinned ncnn patches"):
            print(f"{job_name} must set up Python before applying ncnn patches", file=sys.stderr)
            return 1
    setup_index = ucrt64.index("- name: Set up MSYS2 UCRT64")
    patch_index = ucrt64.index("- name: Apply pinned ncnn patches")
    if setup_index > patch_index:
        print("UCRT64 must set up the msys2 shell before applying ncnn patches", file=sys.stderr)
        return 1
    patch_end = ucrt64.index("\n      - name:", patch_index + 1)
    if "shell: pwsh" not in ucrt64[patch_index:patch_end]:
        print("UCRT64 ncnn patching must use the host PowerShell Python", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
