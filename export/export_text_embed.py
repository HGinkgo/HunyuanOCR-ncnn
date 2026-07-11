#!/usr/bin/env python3
"""Export HunyuanOCR token embedding to ncnn."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import HunYuanVLForConditionalGeneration

from _common import ensure_dir, input_embedding, run_pnnx


class TextEmbedWrapper(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.embed_tokens = nn.Embedding.from_pretrained(weight.detach().clone().float(), freeze=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids.to(torch.int32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HunyuanOCR token embedding.")
    parser.add_argument("--hf-dir", type=Path, required=True, help="HunyuanOCR HuggingFace model directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("models/export/text_embed"))
    parser.add_argument("--seq-len", type=int, default=32, help="Trace sequence length. Default: 32")
    parser.add_argument("--pnnx", type=Path, default=None, help="pnnx executable. Required unless --skip-pnnx is used.")
    parser.add_argument("--skip-pnnx", action="store_true", help="Only write TorchScript .pt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        str(args.hf_dir),
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).eval()
    wrapper = TextEmbedWrapper(input_embedding(model).weight).eval()
    input_ids = torch.arange(args.seq_len, dtype=torch.int32).view(1, args.seq_len)
    pt_path = out_dir / "text_embed.pt"
    torch.jit.trace(wrapper, input_ids).save(str(pt_path))
    print("saved:", pt_path)

    if not args.skip_pnnx:
        if args.pnnx is None:
            raise SystemExit("--pnnx is required unless --skip-pnnx is used")
        run_pnnx(
            args.pnnx,
            pt_path,
            out_dir / "text_embed.ncnn.param",
            out_dir / "text_embed.ncnn.bin",
            f"[1,{args.seq_len}]i32",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
