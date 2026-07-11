from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

try:
    from _common import vision_tower
except ModuleNotFoundError:
    from export._common import vision_tower


DEFAULT_IMAGE_DIR = Path("examples/images")
DEFAULT_BASELINE_ROOT = Path("outputs/baseline_fp32_p512k")
DEFAULT_OUT_DIR = Path("models/export/vision_dynamic_probe")
_NCNN_BINDING: Any | None = None

SPOTTING_PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"
DOCUMENT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


@dataclass(frozen=True)
class CaseSpec:
    image_name: str
    prompt_kind: str

    @property
    def stem(self) -> str:
        return Path(self.image_name).stem

    @property
    def prompt(self) -> str:
        return SPOTTING_PROMPT if self.prompt_kind == "spotting" else DOCUMENT_PROMPT


CASES = [
    CaseSpec("hf_demo_tools-dark.png", "spotting"),
    CaseSpec("omnidoc_document_zh_page-205e4273-5b94-43e5-bfaf-dc882416b067.png", "spotting"),
    CaseSpec("omnidoc_document_book_docstructbench_enbook_19221575_1173.jpg", "document"),
    CaseSpec("omnidoc_table_pyomo_page_188.png", "document"),
    CaseSpec("omnidoc_formula_harmonic_analysis_page_119.png", "document"),
]


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size: int = 1152, intermediate_size: int = 4304):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))


class VisionAttention(nn.Module):
    def __init__(self, hidden_size: int = 1152, num_heads: int = 16, head_dim: int = 72):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class VisionBlock(nn.Module):
    def __init__(self, hidden_size: int = 1152, intermediate_size: int = 4304, num_heads: int = 16, head_dim: int = 72):
        super().__init__()
        self.self_attn = VisionAttention(hidden_size, num_heads, head_dim)
        self.mlp = VisionMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class VisionPatchMerger(nn.Module):
    def __init__(self, in_channels: int = 1152, out_channels: int = 1024, merge_size: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=merge_size, stride=merge_size),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(in_channels * 4, out_channels, bias=True)
        self.image_newline = nn.Parameter(torch.zeros(in_channels * 4))
        self.image_begin = nn.Parameter(torch.zeros(out_channels))
        self.image_end = nn.Parameter(torch.zeros(out_channels))
        self.before_rms = RMSNorm(in_channels, eps=1e-5)
        self.after_rms = RMSNorm(out_channels, eps=1e-5)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        x = self.before_rms(x)
        x = x.permute(0, 2, 1).reshape(1, self.in_channels, grid_h, grid_w)
        x = self.proj(x)
        c = x.size(1)
        newline = self.image_newline.reshape(1, c, 1, 1) + x[:, :1, :, :1] * 0.0
        x = torch.cat([x, newline], dim=3)
        x = x.reshape(1, c, -1).permute(0, 2, 1)
        x = self.mlp(x)
        begin = self.image_begin.reshape(1, 1, self.out_channels)
        end = self.image_end.reshape(1, 1, self.out_channels)
        x = torch.cat([begin, x, end], dim=1)
        return self.after_rms(x)


class DynamicVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1152
        self.patch_embedding = nn.Conv2d(3, 1152, kernel_size=16, stride=16, bias=True)
        self.layers = nn.ModuleList([VisionBlock() for _ in range(27)])
        self.perceive = VisionPatchMerger()

    def forward(self, pixels: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(pixels)
        grid_h = x.size(2)
        grid_w = x.size(3)
        x = x + pos_embed
        x = x.reshape(1, self.hidden_size, grid_h * grid_w).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        return self.perceive(x, grid_h, grid_w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["torch", "export", "ncnn", "token", "all"], default="torch")
    parser.add_argument("--hf-dir", type=Path, required=True, help="HunyuanOCR HuggingFace model directory.")
    parser.add_argument("--pnnx", type=Path, default=None, help="pnnx executable. Required for --mode export/all.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ncnn-python-dir", type=Path, default=None, help="Directory containing the ncnn Python binding.")
    parser.add_argument("--case", action="append", default=[], help="case stem or image filename; repeatable")
    parser.add_argument("--pos-method", choices=["scale", "size", "both"], default="size")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--export-h1", type=int, default=384)
    parser.add_argument("--export-w1", type=int, default=896)
    parser.add_argument("--export-h2", type=int, default=224)
    parser.add_argument("--export-w2", type=int, default=448)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    return parser.parse_args()


def copy_linear(dst: nn.Linear, src: nn.Linear, bias: bool = True) -> None:
    dst.weight.data.copy_(src.weight.data)
    if bias:
        dst.bias.data.copy_(src.bias.data)


def copy_conv(dst: nn.Conv2d, src: nn.Conv2d) -> None:
    dst.weight.data.copy_(src.weight.data)
    dst.bias.data.copy_(src.bias.data)


@torch.no_grad()
def build_dynamic_vision(vit: nn.Module) -> DynamicVisionEncoder:
    wrapper = DynamicVisionEncoder().eval()
    copy_conv(wrapper.patch_embedding, vit.embeddings.patch_embedding)
    for dst, src in zip(wrapper.layers, vit.layers):
        copy_linear(dst.self_attn.q_proj, src.self_attn.q_proj)
        copy_linear(dst.self_attn.k_proj, src.self_attn.k_proj)
        copy_linear(dst.self_attn.v_proj, src.self_attn.v_proj)
        copy_linear(dst.self_attn.o_proj, src.self_attn.o_proj)
        copy_linear(dst.mlp.dense_h_to_4h, getattr(src.mlp, "fc1", src.mlp.dense_h_to_4h if hasattr(src.mlp, "dense_h_to_4h") else None))
        copy_linear(dst.mlp.dense_4h_to_h, getattr(src.mlp, "fc2", src.mlp.dense_4h_to_h if hasattr(src.mlp, "dense_4h_to_h") else None))
        input_norm = getattr(src, "layer_norm1", src.input_layernorm if hasattr(src, "input_layernorm") else None)
        output_norm = getattr(src, "layer_norm2", src.post_attention_layernorm if hasattr(src, "post_attention_layernorm") else None)
        dst.input_layernorm.weight.data.copy_(input_norm.weight.data)
        dst.input_layernorm.bias.data.copy_(input_norm.bias.data)
        dst.post_attention_layernorm.weight.data.copy_(output_norm.weight.data)
        dst.post_attention_layernorm.bias.data.copy_(output_norm.bias.data)
    merger = getattr(vit, "patch_merger", vit.perceive if hasattr(vit, "perceive") else None)
    proj_conv = getattr(merger, "proj_conv", merger.proj[0] if hasattr(merger, "proj") else None)
    proj_out = getattr(merger, "proj_out", merger.proj[2] if hasattr(merger, "proj") else None)
    copy_conv(wrapper.perceive.proj[0], proj_conv)
    copy_conv(wrapper.perceive.proj[2], proj_out)
    copy_linear(wrapper.perceive.mlp, merger.mlp)
    wrapper.perceive.image_newline.data.copy_(merger.image_newline.data)
    wrapper.perceive.image_begin.data.copy_(merger.image_begin.data)
    wrapper.perceive.image_end.data.copy_(merger.image_end.data)
    wrapper.perceive.before_rms.weight.data.copy_(merger.before_rms.weight.data)
    wrapper.perceive.after_rms.weight.data.copy_(merger.after_rms.weight.data)
    return wrapper


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def apply_processor_pixel_bounds(processor: AutoProcessor, min_pixels: int, max_pixels: int) -> None:
    image_processor = processor.image_processor
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    image_processor.size = {
        "shortest_edge": int(min_pixels),
        "longest_edge": int(max_pixels),
    }


def selected_cases(names: list[str]) -> list[CaseSpec]:
    if not names:
        names = ["hf_demo_tools-dark", "omnidoc_document_zh_page-205e4273-5b94-43e5-bfaf-dc882416b067"]
    result: list[CaseSpec] = []
    for name in names:
        matches = [c for c in CASES if c.stem == name or c.image_name == name]
        if not matches:
            raise FileNotFoundError(f"case not found: {name}")
        result.append(matches[0])
    return result


def load_model_and_processor(args: argparse.Namespace) -> tuple[AutoProcessor, nn.Module, DynamicVisionEncoder]:
    processor = AutoProcessor.from_pretrained(str(args.hf_dir), use_fast=False)
    apply_processor_pixel_bounds(processor, args.min_pixels, args.max_pixels)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        str(args.hf_dir),
        attn_implementation="eager",
        dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    ).eval()
    vit = vision_tower(model).float().eval()
    wrapper = build_dynamic_vision(vit).eval()
    return processor, vit, wrapper


def load_case_inputs(processor: AutoProcessor, image_dir: Path, case: CaseSpec) -> tuple[torch.Tensor, torch.Tensor]:
    image_path = image_dir / case.image_name
    image = Image.open(image_path)
    messages = build_messages(image_path, case.prompt)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=image, padding=True, return_tensors="pt")
    return inputs["pixel_values"].float(), inputs["image_grid_thw"].to(torch.int64)


def image_from_pixel_values(pixel_values: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    patch = 16
    img = pixel_values.reshape(grid_h, grid_w, 3, patch, patch)
    return img.permute(2, 0, 3, 1, 4).reshape(1, 3, grid_h * patch, grid_w * patch).contiguous()


def base_pos_embed(vit: nn.Module) -> torch.Tensor:
    pos_edge = int((vit.embeddings.position_embedding.weight.shape[0] - 1) ** 0.5)
    hidden = vit.embeddings.position_embedding.weight.shape[1]
    return vit.embeddings.position_embedding.weight[1:, :].reshape(1, pos_edge, pos_edge, hidden).permute(0, 3, 1, 2).float()


def vision_interpolate_mode(vit: nn.Module) -> str:
    if hasattr(vit.embeddings, "interpolate_mode"):
        return vit.embeddings.interpolate_mode
    return vit.embeddings.config.interpolate_mode


def build_pos_embed(vit: nn.Module, grid_h: int, grid_w: int, method: str) -> torch.Tensor:
    pos_base = base_pos_embed(vit)
    pos_edge = pos_base.shape[-1]
    if method == "scale":
        return F.interpolate(
            pos_base,
            scale_factor=((grid_h + 0.1) / pos_edge, (grid_w + 0.1) / pos_edge),
            mode=vision_interpolate_mode(vit),
            align_corners=False,
        ).contiguous()
    if method == "size":
        return F.interpolate(
            pos_base,
            size=[grid_h, grid_w],
            mode=vision_interpolate_mode(vit),
            align_corners=False,
        ).contiguous()
    raise ValueError(method)


def diff_stats(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = np.abs(actual.astype(np.float32, copy=False) - expected.astype(np.float32, copy=False))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.quantile(diff.reshape(-1), 0.99)),
    }


def generation_options(max_new_tokens: int, repetition_penalty: float) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": repetition_penalty,
    }


@torch.no_grad()
def run_torch(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods = ["scale", "size"] if args.pos_method == "both" else [args.pos_method]
    processor, vit, wrapper = load_model_and_processor(args)
    device = torch.device(args.device)
    vit = vit.to(device)
    wrapper = wrapper.to(device)

    summary: dict[str, Any] = {
        "mode": "torch",
        "baseline_root": str(args.baseline_root),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "pos_methods": methods,
        "cases": [],
    }
    for case in selected_cases(args.case):
        pixel_values, image_grid_thw = load_case_inputs(processor, args.image_dir, case)
        grid_t, grid_h, grid_w = [int(x) for x in image_grid_thw.reshape(-1).tolist()]
        if grid_t != 1:
            raise RuntimeError(f"only grid_t=1 is supported in this probe, got {image_grid_thw.tolist()}")

        baseline_case = args.baseline_root / case.stem
        with np.load(baseline_case / "vision_features.npz") as data:
            baseline_features = data["vision_features"].astype(np.float32, copy=False)
            baseline_grid = data["image_grid_thw"].astype(np.int64, copy=False)
        if image_grid_thw.numpy().astype(np.int64).tolist() != baseline_grid.tolist():
            raise RuntimeError(f"grid mismatch for {case.stem}: processor={image_grid_thw.tolist()} baseline={baseline_grid.tolist()}")

        pixel_values_d = pixel_values.to(device)
        grid_d = image_grid_thw.to(device)
        hf_output = vit(pixel_values_d, grid_d)
        hf_features = getattr(hf_output, "pooler_output", hf_output).float().cpu().numpy()
        image_chw = image_from_pixel_values(pixel_values, grid_h, grid_w)

        case_result: dict[str, Any] = {
            "case": case.stem,
            "image_grid_thw": image_grid_thw.tolist(),
            "image_chw_shape": list(image_chw.shape),
            "baseline_shape": list(baseline_features.shape),
            "hf_vs_baseline": diff_stats(hf_features, baseline_features),
            "methods": {},
        }
        case_dir = args.out_dir / "torch" / case.stem
        case_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            pos = build_pos_embed(vit, grid_h, grid_w, method).to(device)
            actual = wrapper(image_chw.to(device), pos).float().cpu().numpy()
            if actual.shape != baseline_features.shape:
                raise RuntimeError(f"{case.stem} {method} shape mismatch: {actual.shape} vs {baseline_features.shape}")
            case_result["methods"][method] = {
                "wrapper_vs_hf": diff_stats(actual, hf_features),
                "wrapper_vs_baseline": diff_stats(actual, baseline_features),
                "pos_embed_shape": list(pos.shape),
            }
            np.savez_compressed(
                case_dir / f"{method}_io.npz",
                image_chw=image_chw.numpy().astype(np.float32, copy=False),
                pos_embed=pos.cpu().numpy().astype(np.float32, copy=False),
                wrapper_features=actual.astype(np.float32, copy=False),
                hf_features=hf_features.astype(np.float32, copy=False),
                baseline_features=baseline_features,
                image_grid_thw=image_grid_thw.numpy().astype(np.int64, copy=False),
            )
            print(json.dumps({"case": case.stem, "method": method, **case_result["methods"][method]}, ensure_ascii=False), flush=True)

        summary["cases"].append(case_result)
        gc.collect()

    summary["elapsed_sec"] = round(time.time() - started, 3)
    out = args.out_dir / "torch_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", out, flush=True)
    return summary


def set_fp32_options(net: Any) -> None:
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    net.opt.use_bf16_packed = False
    net.opt.use_packing_layout = False
    net.opt.use_vulkan_compute = False


def import_ncnn_binding() -> Any:
    global _NCNN_BINDING
    if _NCNN_BINDING is not None and hasattr(_NCNN_BINDING, "Net"):
        return _NCNN_BINDING

    binding_dir = getattr(import_ncnn_binding, "binding_dir", None)
    if binding_dir is None:
        raise RuntimeError("--ncnn-python-dir is required for ncnn comparison modes")
    existing = sys.modules.get("ncnn")
    if existing is not None and not hasattr(existing, "Net"):
        sys.modules.pop("ncnn", None)
    sys.path.insert(0, str(binding_dir))
    import ncnn

    if not hasattr(ncnn, "Net"):
        raise RuntimeError(f"failed to import ncnn python binding from {binding_dir}")
    _NCNN_BINDING = ncnn
    return ncnn


@torch.no_grad()
def run_export(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, vit, wrapper = load_model_and_processor(args)
    wrapper = wrapper.cpu().eval()
    export_dir = args.out_dir / "ncnn"
    export_dir.mkdir(parents=True, exist_ok=True)
    pt_path = export_dir / "vision_dynamic.pt"
    param_path = export_dir / "vision_dynamic.ncnn.param"
    bin_path = export_dir / "vision_dynamic.ncnn.bin"
    pos_embed_path = export_dir / "pos_embed.bin"

    pos1 = torch.zeros(1, 1152, args.export_h1 // 16, args.export_w1 // 16, dtype=torch.float32)
    pos2 = torch.zeros(1, 1152, args.export_h2 // 16, args.export_w2 // 16, dtype=torch.float32)
    x1 = torch.randn(1, 3, args.export_h1, args.export_w1, dtype=torch.float32)
    traced = torch.jit.trace(wrapper, (x1, pos1), check_trace=False)
    traced.save(str(pt_path))
    del traced
    gc.collect()

    cmd = [
        str(args.pnnx),
        str(pt_path),
        f"ncnnparam={param_path}",
        f"ncnnbin={bin_path}",
        f"inputshape=[1,3,{args.export_h1},{args.export_w1}],[1,1152,{args.export_h1 // 16},{args.export_w1 // 16}]",
        f"inputshape2=[1,3,{args.export_h2},{args.export_w2}],[1,1152,{args.export_h2 // 16},{args.export_w2 // 16}]",
        "fp16=0",
        "optlevel=2",
    ]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=export_dir, check=True)
    pos_embed = base_pos_embed(vit).cpu().numpy().astype(np.float32, copy=False)
    pos_embed_path.write_bytes(pos_embed.tobytes())
    summary = {
        "mode": "export",
        "pt": str(pt_path),
        "param": str(param_path),
        "bin": str(bin_path),
        "pos_embed": str(pos_embed_path),
        "inputshape": [[1, 3, args.export_h1, args.export_w1], [1, 1152, args.export_h1 // 16, args.export_w1 // 16]],
        "inputshape2": [[1, 3, args.export_h2, args.export_w2], [1, 1152, args.export_h2 // 16, args.export_w2 // 16]],
        "elapsed_sec": round(time.time() - started, 3),
    }
    out = args.out_dir / "export_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", out, flush=True)
    return summary


def run_ncnn_case(param_path: Path, bin_path: Path, image_chw: np.ndarray, pos_embed: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    ncnn = import_ncnn_binding()

    with ncnn.Net() as net:
        set_fp32_options(net)
        if net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"load param failed: {param_path}")
        if net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"load bin failed: {bin_path}")
        with net.create_extractor() as ex:
            if ex.input("in0", ncnn.Mat(image_chw.squeeze(0)).clone()) != 0:
                raise RuntimeError("input in0 failed")
            if ex.input("in1", ncnn.Mat(pos_embed.squeeze(0)).clone()) != 0:
                raise RuntimeError("input in1 failed")
            ret, out0 = ex.extract("out0")
            if ret != 0:
                raise RuntimeError(f"extract out0 failed: {ret}")
            return np.array(out0).astype(np.float32, copy=False).reshape(expected_shape)


def run_ncnn_compare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    methods = ["scale", "size"] if args.pos_method == "both" else [args.pos_method]
    param_path = args.out_dir / "ncnn" / "vision_dynamic.ncnn.param"
    bin_path = args.out_dir / "ncnn" / "vision_dynamic.ncnn.bin"
    if not param_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"missing ncnn files under {param_path.parent}")

    summary: dict[str, Any] = {
        "mode": "ncnn",
        "param": str(param_path),
        "bin": str(bin_path),
        "pos_methods": methods,
        "cases": [],
    }
    for case in selected_cases(args.case):
        case_dir = args.out_dir / "torch" / case.stem
        case_result: dict[str, Any] = {"case": case.stem, "methods": {}}
        for method in methods:
            npz_path = case_dir / f"{method}_io.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"missing torch io: {npz_path}; run --mode torch first")
            with np.load(npz_path) as data:
                image_chw = data["image_chw"].astype(np.float32, copy=False)
                pos_embed = data["pos_embed"].astype(np.float32, copy=False)
                wrapper_features = data["wrapper_features"].astype(np.float32, copy=False)
                baseline_features = data["baseline_features"].astype(np.float32, copy=False)
            actual = run_ncnn_case(param_path, bin_path, image_chw, pos_embed, tuple(wrapper_features.shape))
            case_result["methods"][method] = {
                "ncnn_vs_wrapper": diff_stats(actual, wrapper_features),
                "ncnn_vs_baseline": diff_stats(actual, baseline_features),
                "actual_shape": list(actual.shape),
            }
            out_dir = args.out_dir / "ncnn_features" / case.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / f"{method}_ncnn_features.npz", vision_features=actual)
            print(json.dumps({"case": case.stem, "method": method, **case_result["methods"][method]}, ensure_ascii=False), flush=True)
        summary["cases"].append(case_result)

    summary["elapsed_sec"] = round(time.time() - started, 3)
    out = args.out_dir / "ncnn_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", out, flush=True)
    return summary


def to_numpy_int64(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.int64).numpy()


@torch.no_grad()
def run_token_gate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.pos_method == "both":
        raise ValueError("--mode token requires --pos-method scale or size")

    processor = AutoProcessor.from_pretrained(str(args.hf_dir), use_fast=False)
    apply_processor_pixel_bounds(processor, args.min_pixels, args.max_pixels)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        str(args.hf_dir),
        attn_implementation="eager",
        dtype=torch.float32,
        device_map="auto",
    ).eval()

    summary: dict[str, Any] = {
        "mode": "token",
        "pos_method": args.pos_method,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "cases": [],
    }
    out_root = args.out_dir / "token_gate"
    out_root.mkdir(parents=True, exist_ok=True)

    for case in selected_cases(args.case):
        image_path = args.image_dir / case.image_name
        image = Image.open(image_path)
        messages = build_messages(image_path, case.prompt)
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=image, padding=True, return_tensors="pt")

        feature_path = args.out_dir / "ncnn_features" / case.stem / f"{args.pos_method}_ncnn_features.npz"
        if not feature_path.is_file():
            raise FileNotFoundError(f"missing dynamic ncnn features: {feature_path}")
        with np.load(feature_path) as data:
            dynamic_features_np = data["vision_features"].astype(np.float32, copy=False)

        baseline_dir = args.baseline_root / case.stem
        with np.load(baseline_dir / "generated_ids.npz") as data:
            expected_generated = data["generated_ids"].astype(np.int64, copy=False)
            expected_trimmed = data["generated_ids_trimmed"].astype(np.int64, copy=False)
        expected_text = (baseline_dir / "output_text.txt").read_text(encoding="utf-8")

        device = next(model.parameters()).device
        inputs = inputs.to(device)
        inputs_embeds = model.model.embed_tokens(inputs["input_ids"])
        image_embeds = torch.from_numpy(dynamic_features_np).to(device=device, dtype=torch.float32)
        image_mask, _ = model.get_placeholder_mask(
            inputs["input_ids"],
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        first_outputs = model(
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            inputs_embeds=inputs_embeds,
            logits_to_keep=1,
            use_cache=True,
        )
        generated_ids = super(HunYuanVLForConditionalGeneration, model).generate(
            inputs=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            inputs_embeds=inputs_embeds,
            **generation_options(args.max_new_tokens, args.repetition_penalty),
        )
        generated_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        output_text = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        generated_np = to_numpy_int64(generated_ids)
        trimmed_np = to_numpy_int64(generated_trimmed)
        match_generated = bool(np.array_equal(generated_np, expected_generated))
        match_trimmed = bool(np.array_equal(trimmed_np, expected_trimmed))
        match_text = output_text == expected_text
        first_mismatch = -1
        if not match_generated:
            min_len = min(generated_np.shape[1], expected_generated.shape[1])
            mismatch = np.nonzero(generated_np[:, :min_len] != expected_generated[:, :min_len])
            if len(mismatch[1]) > 0:
                first_mismatch = int(mismatch[1][0])
            elif generated_np.shape[1] != expected_generated.shape[1]:
                first_mismatch = min_len

        case_dir = out_root / case.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / f"{args.pos_method}_generated_ids.npz",
            generated_ids=generated_np,
            generated_ids_trimmed=trimmed_np,
            first_logits=first_outputs.logits.detach().cpu().to(torch.float32).numpy(),
        )
        (case_dir / f"{args.pos_method}_output_text.txt").write_text(output_text, encoding="utf-8")

        item = {
            "case": case.stem,
            "dynamic_features_shape": list(dynamic_features_np.shape),
            "generated_ids_shape": list(generated_np.shape),
            "expected_generated_ids_shape": list(expected_generated.shape),
            "match_generated_ids": match_generated,
            "match_trimmed_ids": match_trimmed,
            "match_text": match_text,
            "first_mismatch_index": first_mismatch,
            "output_text_chars": len(output_text),
        }
        summary["cases"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

        del inputs, inputs_embeds, image_embeds, image_mask, first_outputs, generated_ids, generated_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    summary["all_match_text"] = all(item["match_text"] for item in summary["cases"])
    summary["all_match_generated_ids"] = all(item["match_generated_ids"] for item in summary["cases"])
    summary["elapsed_sec"] = round(time.time() - started, 3)
    out = args.out_dir / "token_gate_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", out, flush=True)
    return summary


def main() -> None:
    args = parse_args()
    args.hf_dir = args.hf_dir.resolve()
    args.image_dir = args.image_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.baseline_root = args.baseline_root.resolve()
    if args.pnnx is not None:
        args.pnnx = args.pnnx.resolve()
    if args.repetition_penalty <= 0:
        raise SystemExit("--repetition-penalty must be positive")
    if args.ncnn_python_dir is not None:
        import_ncnn_binding.binding_dir = args.ncnn_python_dir.resolve()
    if args.mode in ("export", "all") and args.pnnx is None:
        raise SystemExit("--pnnx is required for --mode export/all")
    if args.mode in ("ncnn", "all") and args.ncnn_python_dir is None:
        raise SystemExit("--ncnn-python-dir is required for --mode ncnn/all")
    if args.mode in ("torch", "all"):
        run_torch(args)
    if args.mode in ("export", "all"):
        run_export(args)
    if args.mode in ("ncnn", "all"):
        run_ncnn_compare(args)
    if args.mode in ("token", "all"):
        run_token_gate(args)


if __name__ == "__main__":
    main()
