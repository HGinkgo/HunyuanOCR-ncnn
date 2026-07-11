#!/usr/bin/env python3
"""Export HunyuanOCR tied lm_head to ncnn."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import HunYuanVLForConditionalGeneration

from _common import ensure_dir, input_embedding, run_pnnx


class LMHeadWrapper(nn.Module):
    def __init__(self, tied_weight: torch.Tensor):
        super().__init__()
        vocab_size, hidden_size = tied_weight.shape
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(tied_weight.detach().clone().float())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states.float())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HunyuanOCR lm_head with tied embedding weight.")
    parser.add_argument("--hf-dir", type=Path, required=True, help="HunyuanOCR HuggingFace model directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("models/export/lm_head"))
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
    wrapper = LMHeadWrapper(input_embedding(model).weight).eval()
    hidden_states = torch.randn(1, 1, wrapper.lm_head.in_features, dtype=torch.float32)
    pt_path = out_dir / "lm_head.pt"
    torch.jit.trace(wrapper, hidden_states).save(str(pt_path))
    print("saved:", pt_path)

    if not args.skip_pnnx:
        if args.pnnx is None:
            raise SystemExit("--pnnx is required unless --skip-pnnx is used")
        run_pnnx(
            args.pnnx,
            pt_path,
            out_dir / "lm_head.ncnn.param",
            out_dir / "lm_head.ncnn.bin",
            "[1,1,1024]f32",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
