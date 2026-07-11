#!/usr/bin/env python3
"""Run the standard HunyuanOCR-ncnn export pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DYNAMIC_VISION_DIR = Path("models/export/vision_dynamic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tokenizer and ncnn submodules from a HunyuanOCR HF model.")
    parser.add_argument("--hf-dir", type=Path, required=True, help="HunyuanOCR HuggingFace model directory.")
    parser.add_argument("--pnnx", type=Path, required=True, help="pnnx executable.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Workspace where models/tokenizer and models/export are written. Default: current directory.",
    )
    parser.add_argument("--skip-text-decoder", action="store_true", help="Skip text decoder export.")
    parser.add_argument("--skip-vision", action="store_true", help="Skip dynamic vision export.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    workspace = args.workspace.resolve()
    hf_dir = args.hf_dir.resolve()
    pnnx = args.pnnx.resolve()

    run(
        [
            sys.executable,
            str(script_dir / "extract_tokenizer.py"),
            "--hf-dir",
            str(hf_dir),
            "--out-dir",
            str(workspace / "models/tokenizer"),
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "export_text_embed.py"),
            "--hf-dir",
            str(hf_dir),
            "--out-dir",
            str(workspace / "models/export/text_embed"),
            "--pnnx",
            str(pnnx),
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "export_lm_head.py"),
            "--hf-dir",
            str(hf_dir),
            "--out-dir",
            str(workspace / "models/export/lm_head"),
            "--pnnx",
            str(pnnx),
        ]
    )
    if not args.skip_text_decoder:
        run(
            [
                sys.executable,
                str(script_dir / "export_text_decoder_kv.py"),
                "--hf-dir",
                str(hf_dir),
                "--out-dir",
                str(workspace / "models/export/text_decoder"),
                "--pnnx",
                str(pnnx),
            ]
        )
    if not args.skip_vision:
        run(
            [
                sys.executable,
                str(script_dir / "export_vision_dynamic.py"),
                "--mode",
                "export",
                "--hf-dir",
                str(hf_dir),
                "--pnnx",
                str(pnnx),
                "--out-dir",
                str(workspace / DEFAULT_DYNAMIC_VISION_DIR),
            ]
        )
    print("export_workspace:", workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
