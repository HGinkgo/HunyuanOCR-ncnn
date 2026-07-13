#!/usr/bin/env python3
"""Export the HunyuanOCR 1.5 DFlash draft to ncnn and compare fp32 outputs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel

from _common import ensure_dir, pnnx_inputshape, pnnx_intermediate_paths, run_pnnx
from dflash_export import DFlashExportWrapper


DEFAULT_OUT_DIR = Path("models/export/dflash")
_NCNN_BINDING: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HunyuanOCR DFlash draft to ncnn.")
    parser.add_argument("--mode", choices=["torch", "export", "ncnn", "all"], default="torch")
    parser.add_argument("--dflash-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pnnx", type=Path, default=None, help="pnnx executable for export/all.")
    parser.add_argument("--ncnn-python-dir", type=Path, default=None)
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--context-len2", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--torch-max-abs", type=float, default=2e-5)
    parser.add_argument("--ncnn-max-abs", type=float, default=1e-4)
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def diff_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = np.abs(actual.astype(np.float32, copy=False) - expected.astype(np.float32, copy=False))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.quantile(diff.reshape(-1), 0.99)),
    }


def load_draft(path: Path) -> torch.nn.Module:
    return AutoModel.from_pretrained(
        str(path),
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).float().eval()


def make_inputs(
    draft: torch.nn.Module,
    context_len: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    block_size = int(draft.block_size)
    hidden_size = int(draft.config.hidden_size)
    noise = torch.randn(1, block_size, hidden_size, generator=generator)
    targets = tuple(
        torch.randn(1, context_len, hidden_size, generator=generator) for _ in range(4)
    )
    position_ids = torch.arange(context_len + block_size, dtype=torch.int64).reshape(1, -1)
    cos, sin = draft.rotary_emb(noise, position_ids)
    attention_mask = torch.zeros(1, 1, block_size, context_len + block_size)
    wrapper_inputs = (noise, *targets, cos.unsqueeze(1), sin.unsqueeze(1), attention_mask)
    return wrapper_inputs, position_ids, torch.cat(targets, dim=-1)


def io_path(out_dir: Path, context_len: int) -> Path:
    return out_dir / "torch" / f"context_{context_len}.npz"


@torch.no_grad()
def run_torch_compare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    draft = load_draft(args.dflash_dir)
    wrapper = DFlashExportWrapper.from_draft(draft).eval()
    summary: dict[str, Any] = {
        "mode": "torch",
        "dflash_dir": str(args.dflash_dir),
        "block_size": int(draft.block_size),
        "target_layer_ids": list(draft.target_layer_ids),
        "cases": [],
    }
    for case_index, context_len in enumerate((args.context_len, args.context_len2)):
        inputs, position_ids, combined_targets = make_inputs(
            draft, context_len, args.seed + case_index
        )
        expected = draft(
            position_ids=position_ids,
            attention_mask=inputs[-1],
            noise_embedding=inputs[0],
            target_hidden=combined_targets,
        )
        actual = wrapper(*inputs)
        expected_np = expected.cpu().numpy().astype(np.float32, copy=False)
        actual_np = actual.cpu().numpy().astype(np.float32, copy=False)
        stats = diff_stats(actual_np, expected_np)
        if stats["max_abs"] > args.torch_max_abs:
            raise RuntimeError(
                f"context {context_len} wrapper max_abs {stats['max_abs']} exceeds {args.torch_max_abs}"
            )
        path = io_path(args.out_dir, context_len)
        ensure_dir(path.parent)
        np.savez_compressed(
            path,
            in0=inputs[0].cpu().numpy(),
            in1=inputs[1].cpu().numpy(),
            in2=inputs[2].cpu().numpy(),
            in3=inputs[3].cpu().numpy(),
            in4=inputs[4].cpu().numpy(),
            in5=inputs[5].cpu().numpy(),
            in6=inputs[6].cpu().numpy(),
            in7=inputs[7].cpu().numpy(),
            wrapper_output=actual_np,
            official_output=expected_np,
        )
        result = {
            "context_len": context_len,
            "output_shape": list(actual_np.shape),
            "wrapper_vs_official": stats,
            "io": str(path),
        }
        summary["cases"].append(result)
        print(json.dumps(result), flush=True)
    summary["elapsed_sec"] = round(time.time() - started, 3)
    path = args.out_dir / "torch_summary.json"
    ensure_dir(path.parent)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return summary


@torch.no_grad()
def run_export(args: argparse.Namespace) -> dict[str, Any]:
    if args.pnnx is None:
        raise ValueError("--pnnx is required for export mode")
    started = time.time()
    draft = load_draft(args.dflash_dir)
    wrapper = DFlashExportWrapper.from_draft(draft).eval()
    block_size = wrapper.block_size
    hidden_size = wrapper.hidden_size
    head_dim = wrapper.head_dim
    inputs, _, _ = make_inputs(draft, args.context_len, args.seed)
    export_dir = ensure_dir(args.out_dir / "ncnn")
    pt_path = export_dir / "dflash.pt"
    param_path = export_dir / "dflash.ncnn.param"
    bin_path = export_dir / "dflash.ncnn.bin"
    traced = torch.jit.trace(wrapper, inputs, strict=False, check_trace=False)
    traced.save(str(pt_path))
    del traced, wrapper, draft
    gc.collect()

    run_pnnx(
        args.pnnx,
        pt_path,
        param_path,
        bin_path,
        pnnx_inputshape(args.context_len, block_size, hidden_size, head_dim),
        pnnx_inputshape(args.context_len2, block_size, hidden_size, head_dim),
    )
    if not args.keep_intermediates:
        for path in pnnx_intermediate_paths(pt_path):
            path.unlink(missing_ok=True)
    summary = {
        "mode": "export",
        "pt": str(pt_path) if args.keep_intermediates else None,
        "param": str(param_path),
        "bin": str(bin_path),
        "intermediates_kept": args.keep_intermediates,
        "inputshape": pnnx_inputshape(args.context_len, block_size, hidden_size, head_dim),
        "inputshape2": pnnx_inputshape(args.context_len2, block_size, hidden_size, head_dim),
        "elapsed_sec": round(time.time() - started, 3),
    }
    path = args.out_dir / "export_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return summary


def set_fp32_options(net: Any) -> None:
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    net.opt.use_bf16_packed = False
    net.opt.use_packing_layout = False
    net.opt.use_vulkan_compute = False


def import_ncnn_binding(binding_dir: Path | None) -> Any:
    global _NCNN_BINDING
    if _NCNN_BINDING is not None:
        return _NCNN_BINDING
    if binding_dir is None:
        raise ValueError("--ncnn-python-dir is required for ncnn mode")
    existing = sys.modules.get("ncnn")
    if existing is not None and not hasattr(existing, "Net"):
        sys.modules.pop("ncnn", None)
    sys.path.insert(0, str(binding_dir))
    import ncnn

    if not hasattr(ncnn, "Net"):
        raise RuntimeError(f"failed to import ncnn binding from {binding_dir}")
    _NCNN_BINDING = ncnn
    return ncnn


def run_ncnn_graph(
    ncnn: Any,
    param_path: Path,
    bin_path: Path,
    inputs: list[np.ndarray],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    with ncnn.Net() as net:
        set_fp32_options(net)
        if net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"load param failed: {param_path}")
        if net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"load model failed: {bin_path}")
        with net.create_extractor() as ex:
            for index, array in enumerate(inputs):
                value = np.ascontiguousarray(array.squeeze(0), dtype=np.float32)
                if ex.input(f"in{index}", ncnn.Mat(value).clone()) != 0:
                    raise RuntimeError(f"input in{index} failed")
            ret, output = ex.extract("out0")
            if ret != 0:
                raise RuntimeError(f"extract out0 failed: {ret}")
            return np.array(output).astype(np.float32, copy=False).reshape(expected_shape)


def run_ncnn_compare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ncnn = import_ncnn_binding(args.ncnn_python_dir)
    param_path = args.out_dir / "ncnn" / "dflash.ncnn.param"
    bin_path = args.out_dir / "ncnn" / "dflash.ncnn.bin"
    summary: dict[str, Any] = {
        "mode": "ncnn",
        "param": str(param_path),
        "bin": str(bin_path),
        "cases": [],
    }
    for context_len in (args.context_len, args.context_len2):
        path = io_path(args.out_dir, context_len)
        if not path.is_file():
            raise FileNotFoundError(f"missing torch IO: {path}; run --mode torch first")
        with np.load(path) as data:
            inputs = [data[f"in{i}"].astype(np.float32, copy=False) for i in range(8)]
            wrapper_output = data["wrapper_output"].astype(np.float32, copy=False)
            official_output = data["official_output"].astype(np.float32, copy=False)
        actual = run_ncnn_graph(
            ncnn, param_path, bin_path, inputs, tuple(wrapper_output.shape)
        )
        wrapper_stats = diff_stats(actual, wrapper_output)
        official_stats = diff_stats(actual, official_output)
        if wrapper_stats["max_abs"] > args.ncnn_max_abs:
            raise RuntimeError(
                f"context {context_len} ncnn max_abs {wrapper_stats['max_abs']} exceeds {args.ncnn_max_abs}"
            )
        result = {
            "context_len": context_len,
            "output_shape": list(actual.shape),
            "ncnn_vs_wrapper": wrapper_stats,
            "ncnn_vs_official": official_stats,
        }
        summary["cases"].append(result)
        print(json.dumps(result), flush=True)
    summary["elapsed_sec"] = round(time.time() - started, 3)
    path = args.out_dir / "ncnn_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return summary


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)
    if args.mode in ("torch", "all"):
        run_torch_compare(args)
    if args.mode in ("export", "all"):
        run_export(args)
    if args.mode in ("ncnn", "all"):
        run_ncnn_compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
