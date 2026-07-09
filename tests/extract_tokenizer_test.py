#!/usr/bin/env python3
"""Tokenizer export behavior tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hunyuanocr_extract_tokenizer_test_") as tmp_text:
        tmp = Path(tmp_text)
        hf_dir = tmp / "hf"
        out_dir = tmp / "out"
        hf_dir.mkdir()

        (hf_dir / "tokenizer.json").write_text(
            json.dumps(
                {
                    "model": {
                        "vocab": {"foo": 0, "bar": 1},
                        "merges": [["f", "oo"], ["b", "ar"]],
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (hf_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "added_tokens_decoder": {
                        "3": {"content": "<special_b>"},
                        "2": {"content": "<special_a>"},
                    },
                    "additional_special_tokens": [],
                    "eos_token_id": 3,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (hf_dir / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [3]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "export/extract_tokenizer.py"),
                "--hf-dir",
                str(hf_dir),
                "--out-dir",
                str(out_dir),
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode

        with (out_dir / "special_tokens.json").open("r", encoding="utf-8") as file:
            special_tokens = json.load(file)
        if not isinstance(special_tokens, list):
            print(f"special_tokens.json must be a JSON array, got {type(special_tokens).__name__}", file=sys.stderr)
            return 1
        if special_tokens != ["<special_a>", "<special_b>"]:
            print(f"unexpected special token order: {special_tokens}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
