#!/usr/bin/env python3
"""Export HunyuanOCR text decoder wrapper to ncnn and patch KV cache I/O."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import HunYuanVLForConditionalGeneration

from _common import ensure_dir, run_pnnx, run_python_script, text_backbone
from text_decoder_kv import TextDecoderExternalRopeKVWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HunyuanOCR text decoder KV model.")
    parser.add_argument("--hf-dir", type=Path, required=True, help="HunyuanOCR HuggingFace model directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("models/export/text_decoder"))
    parser.add_argument("--pnnx", type=Path, default=None, help="pnnx executable. Required unless --skip-pnnx is used.")
    parser.add_argument("--seq-len", type=int, default=1, help="Trace sequence length. Default: 1")
    parser.add_argument(
        "--seq-len2",
        type=int,
        default=32,
        help="Alternative sequence length for pnnx inputshape2. Default: 32",
    )
    parser.add_argument("--skip-pnnx", action="store_true", help="Only write TorchScript .pt.")
    parser.add_argument("--skip-kvcache-patch", action="store_true", help="Do not patch ncnn param with KV cache I/O.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        str(args.hf_dir),
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).float().eval()
    wrapper = TextDecoderExternalRopeKVWrapper.from_decoder(text_backbone(model)).eval()

    seq_len = args.seq_len
    inputs_embeds = torch.randn(1, seq_len, 1024, dtype=torch.float32)
    attention_mask = torch.zeros((1, seq_len), dtype=torch.float32)
    xd_cos = torch.ones(1, 1, seq_len, 128, dtype=torch.float32)
    xd_sin = torch.zeros(1, 1, seq_len, 128, dtype=torch.float32)
    rope_cos = torch.ones(1, 1, seq_len, 128, dtype=torch.float32)
    rope_sin = torch.zeros(1, 1, seq_len, 128, dtype=torch.float32)

    pt_path = out_dir / "text_decoder_kv.pt"
    traced = torch.jit.trace(
        wrapper,
        (inputs_embeds, attention_mask, xd_cos, xd_sin, rope_cos, rope_sin),
        strict=False,
    )
    traced.save(str(pt_path))
    print("saved:", pt_path)

    if not args.skip_pnnx:
        if args.pnnx is None:
            raise SystemExit("--pnnx is required unless --skip-pnnx is used")
        param_path = out_dir / "text_decoder_kv.ncnn.param"
        run_pnnx(
            args.pnnx,
            pt_path,
            param_path,
            out_dir / "text_decoder_kv.ncnn.bin",
            f"[1,{seq_len},1024]f32,[1,{seq_len}]f32,[1,1,{seq_len},128]f32,[1,1,{seq_len},128]f32,[1,1,{seq_len},128]f32,[1,1,{seq_len},128]f32",
            f"[1,{args.seq_len2},1024]f32,[1,{args.seq_len2}]f32,[1,1,{args.seq_len2},128]f32,[1,1,{args.seq_len2},128]f32,[1,1,{args.seq_len2},128]f32,[1,1,{args.seq_len2},128]f32",
        )
        if not args.skip_kvcache_patch:
            run_python_script(Path(__file__).with_name("add_kvcache.py"), "--param", str(param_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
