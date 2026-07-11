#!/usr/bin/env python3

from __future__ import annotations

import sys
import shutil
import subprocess
import tempfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_cli_check(root: Path, binary: Path) -> None:
    required_files = (
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
    with tempfile.TemporaryDirectory(prefix="hunyuan_utf8_混元_U0001f600_") as temporary:
        temporary_root = Path(temporary)
        model_root = temporary_root / "模型_U0001f600"
        for relative_path in required_files:
            target = model_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

        image_path = temporary_root / "图片_U0001f600.png"
        shutil.copyfile(root / "examples/images/hf_demo_tools-dark.png", image_path)
        completed = subprocess.run(
            [str(binary), "--model", str(model_root), "--image", str(image_path)],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        require(completed.returncode == 0, completed.stdout + completed.stderr)
        require(f"Model root: {model_root}" in completed.stdout, "Unicode model path output mismatch")
        require(f"  path: {image_path}" in completed.stdout, "Unicode image path output mismatch")

        prompt = "只输出可见文字U0001f600"
        prompt_parse = subprocess.run(
            [str(binary), "--model", str(model_root), "--prompt", prompt],
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        require(prompt_parse.returncode == 0, prompt_parse.stdout + prompt_parse.stderr)


def main() -> int:
    require(len(sys.argv) == 2, "usage: windows_utf8_integration_test.py /path/to/hunyuan_ocr_cli")
    root = Path(__file__).resolve().parents[1]
    binary = Path(sys.argv[1]).resolve()
    main_cpp = (root / "src/main.cpp").read_text(encoding="utf-8")
    image_cpp = (root / "src/image_preprocessor.cpp").read_text(encoding="utf-8")
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")

    for symbol in ("GetCommandLineW", "CommandLineToArgvW", "SetConsoleOutputCP", "wide_arguments_to_utf8"):
        require(symbol in main_cpp, f"Windows UTF-8 CLI integration missing {symbol}")
    require("STBI_WINDOWS_UTF8" in image_cpp, "stb_image UTF-8 filename support is not enabled")
    require("shell32" in cmake.lower(), "Windows CLI must link shell32 for CommandLineToArgvW")

    path_sources = [
        root / "src/generation_config.cpp",
        root / "src/image_preprocessor.cpp",
        root / "src/main.cpp",
        root / "src/model_layout.cpp",
        root / "src/text_runtime.cpp",
        root / "src/tokenizer.cpp",
        root / "src/vision_runtime.cpp",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in path_sources)
    require("path_from_utf8" in combined, "UTF-8 path boundary helper is not used")
    require(".string().c_str()" not in combined, "ncnn path loading must use native filesystem path strings")
    for stale in (
        "std::filesystem::path root(model_root)",
        "std::filesystem::path root(fixture_dir)",
        "std::filesystem::path root_path(model_root)",
    ):
        require(stale not in combined, f"external UTF-8 path still uses the locale constructor: {stale}")
    run_cli_check(root, binary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
