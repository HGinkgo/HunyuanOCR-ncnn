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
