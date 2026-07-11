#!/usr/bin/env python3
"""Dynamic vision export behavior tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace


class _DummyTraced:
    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, destination: str) -> None:
        Path(destination).write_bytes(b"dummy traced module")


def main() -> int:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        print(f"SKIP export_vision_dynamic_test: missing optional export dependency: {exc.name}")
        return 0

    class _DummyWrapper(nn.Module):
        def forward(self, pixels: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
            return pixels.mean() + pos_embed.mean()

    class _FakeVit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight = torch.arange((4 * 4 + 1) * 8, dtype=torch.float32).reshape(17, 8)
            self.embeddings = SimpleNamespace(position_embedding=SimpleNamespace(weight=weight))

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    pil_module = types.ModuleType("PIL")
    pil_module.Image = object()
    sys.modules.setdefault("PIL", pil_module)

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoProcessor = object
    transformers_module.HunYuanVLForConditionalGeneration = object
    sys.modules.setdefault("transformers", transformers_module)

    from export import export_vision_dynamic as mod

    if mod.DEFAULT_OUT_DIR != Path("models/export/vision_dynamic"):
        print(f"dynamic vision default output path mismatch: {mod.DEFAULT_OUT_DIR}", file=sys.stderr)
        return 1

    original_argv = sys.argv
    try:
        sys.argv = ["export_vision_dynamic.py", "--hf-dir", "/tmp/hunyuanocr-1.5"]
        parsed = mod.parse_args()
        if parsed.repetition_penalty != 1.08:
            print("dynamic vision export does not default to repetition penalty 1.08", file=sys.stderr)
            return 1
        if parsed.pos_method != "size":
            print("dynamic vision export does not default to Transformers 5.13 size interpolation", file=sys.stderr)
            return 1
    finally:
        sys.argv = original_argv

    expected_generation_options = {
        "max_new_tokens": 128,
        "do_sample": False,
        "repetition_penalty": 1.08,
    }
    if mod.generation_options(128, 1.08) != expected_generation_options:
        print("generation options mismatch", file=sys.stderr)
        return 1

    new_embeddings = SimpleNamespace(config=SimpleNamespace(interpolate_mode="bilinear"))
    if mod.vision_interpolate_mode(SimpleNamespace(embeddings=new_embeddings)) != "bilinear":
        print("failed to read Transformers 5.13 vision interpolation mode", file=sys.stderr)
        return 1
    legacy_embeddings = SimpleNamespace(interpolate_mode="bilinear")
    if mod.vision_interpolate_mode(SimpleNamespace(embeddings=legacy_embeddings)) != "bilinear":
        print("failed to keep legacy vision interpolation mode fallback", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="hunyuanocr_export_vision_test_") as tmp_text:
        tmp = Path(tmp_text)
        out_dir = tmp / "out"
        args = SimpleNamespace(
            out_dir=out_dir,
            hf_dir=tmp / "hf",
            pnnx=tmp / "pnnx",
            export_h1=64,
            export_w1=96,
            export_h2=32,
            export_w2=48,
        )

        original_loader = mod.load_model_and_processor
        original_trace = torch.jit.trace
        original_run = subprocess.run
        try:
            mod.load_model_and_processor = lambda _args: (None, _FakeVit(), _DummyWrapper())
            torch.jit.trace = lambda wrapper, inputs, check_trace=False: _DummyTraced(out_dir / "ncnn" / "vision_dynamic.pt")
            subprocess.run = lambda *cmd, **kwargs: SimpleNamespace(returncode=0)
            mod.run_export(args)
        finally:
            mod.load_model_and_processor = original_loader
            torch.jit.trace = original_trace
            subprocess.run = original_run

        pos_embed = out_dir / "ncnn" / "pos_embed.bin"
        if not pos_embed.is_file():
            print(f"missing pos_embed.bin: {pos_embed}", file=sys.stderr)
            return 1

        actual = np.frombuffer(pos_embed.read_bytes(), dtype=np.float32).copy()
        expected = mod.base_pos_embed(_FakeVit()).cpu().numpy().reshape(-1)
        if actual.size != expected.size or not np.array_equal(actual, expected):
            print("pos_embed.bin content mismatch", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
