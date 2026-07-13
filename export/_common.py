#!/usr/bin/env python3
"""Shared helpers for HunyuanOCR-ncnn export scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def text_backbone(model: Any) -> Any:
    return getattr(model.model, "language_model", model.model)


def input_embedding(model: Any) -> Any:
    return text_backbone(model).embed_tokens


def vision_tower(model: Any) -> Any:
    return getattr(model.model, "vision_tower", model.vit if hasattr(model, "vit") else None)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def pnnx_inputshape(
    context_len: int,
    block_size: int,
    hidden_size: int,
    head_dim: int,
) -> str:
    total_len = context_len + block_size
    shapes = [f"[1,{block_size},{hidden_size}]f32"]
    shapes.extend([f"[1,{context_len},{hidden_size}]f32"] * 4)
    shapes.extend(
        [
            f"[1,1,{total_len},{head_dim}]f32",
            f"[1,1,{total_len},{head_dim}]f32",
            f"[1,1,{block_size},{total_len}]f32",
        ]
    )
    return ",".join(shapes)


def pnnx_intermediate_paths(pt_path: Path) -> list[Path]:
    stem = pt_path.with_suffix("")
    return [
        pt_path,
        stem.with_suffix(".pnnx.param"),
        stem.with_suffix(".pnnx.bin"),
        stem.with_suffix(".pnnx.onnx"),
        stem.with_name(stem.name + "_pnnx.py"),
        stem.with_name(stem.name + "_ncnn.py"),
    ]


def run_pnnx(
    pnnx: Path,
    pt_path: Path,
    param_path: Path,
    bin_path: Path,
    inputshape: str | None,
    inputshape2: str | None = None,
    extra_args: list[str] | None = None,
) -> None:
    require_file(pnnx, "pnnx executable")
    require_file(pt_path, "TorchScript model")
    pnnx = pnnx.resolve()
    pt_path = pt_path.resolve()
    param_path = param_path.resolve()
    bin_path = bin_path.resolve()
    cmd = [
        str(pnnx),
        str(pt_path),
    ]
    if inputshape:
        cmd.append(f"inputshape={inputshape}")
    if inputshape2:
        cmd.append(f"inputshape2={inputshape2}")
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(
        [
            f"ncnnparam={param_path}",
            f"ncnnbin={bin_path}",
            "fp16=0",
            "optlevel=2",
        ]
    )
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=pt_path.parent, check=True)
