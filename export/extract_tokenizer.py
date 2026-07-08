#!/usr/bin/env python3
"""Export HunyuanOCR tokenizer files used by the C++ runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ensure_dir, require_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract tokenizer files from a HunyuanOCR HF model directory.")
    parser.add_argument("--hf-dir", type=Path, required=True, help="HunyuanOCR HuggingFace model directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/tokenizer"),
        help="Output directory for vocab/merges/special tokens. Default: models/tokenizer",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    require_file(path, path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    tokenizer_json = load_json(args.hf_dir / "tokenizer.json")
    tokenizer_config = load_json(args.hf_dir / "tokenizer_config.json")
    generation_config = load_json(args.hf_dir / "generation_config.json")

    model = tokenizer_json.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])
    added_tokens = tokenizer_config.get("added_tokens_decoder", {})

    max_id = max(vocab.values())
    for token_id in added_tokens:
        max_id = max(max_id, int(token_id))
    id_to_token = [None] * (max_id + 1)
    for token, token_id in vocab.items():
        id_to_token[int(token_id)] = token
    for token_id, spec in added_tokens.items():
        id_to_token[int(token_id)] = spec["content"]
    for token_id, token in enumerate(id_to_token):
        if token is None:
            id_to_token[token_id] = f"<|unused_{token_id}|>"

    (out_dir / "vocab.txt").write_text("\n".join(id_to_token) + "\n", encoding="utf-8")
    merge_lines: list[str] = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            merge_lines.append(f"{merge[0]} {merge[1]}")
        elif isinstance(merge, str):
            merge_lines.append(merge)
    (out_dir / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

    special_tokens = {
        "additional_special_tokens": tokenizer_config.get("additional_special_tokens", []),
        "added_tokens_decoder": added_tokens,
    }
    (out_dir / "special_tokens.json").write_text(
        json.dumps(special_tokens, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    eos_ids = generation_config.get("eos_token_id", tokenizer_config.get("eos_token_id", []))
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    (out_dir / "eos_ids.json").write_text(
        json.dumps({"eos_ids": eos_ids}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("tokenizer_out:", out_dir)
    print("vocab_size:", len(id_to_token))
    print("merges:", len(merge_lines))
    print("eos_ids:", eos_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
