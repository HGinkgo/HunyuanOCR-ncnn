#!/usr/bin/env python3
"""Run a deterministic HunyuanOCR Transformers fp32 baseline for a manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from atomic_output import build_directory_transactionally, lexical_absolute_path, paths_overlap, validate_case_names


SPOTTING_PROMPT = "检测并识别图片中的文字，将文本坐标格式化输出。"
DOCUMENT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)

np = None
torch = None
transformers = None
Image = None
AutoProcessor = None
HunYuanVLForConditionalGeneration = None

FIXED_MODEL_REVISION = "9e01f897bf8956f77a80c350dc0491d6bbbd43e6"
FIXED_TRANSFORMERS_VERSION = "5.13.0"


@dataclass(frozen=True)
class BaselineCase:
    name: str
    image: str
    prompt: str
    prompt_label: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    parser = argparse.ArgumentParser(description="Run HunyuanOCR fp32 baseline for a regression manifest.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="HuggingFace HunyuanOCR model directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "examples/custom_prompt_cases.json",
        help="Manifest with image plus prompt_mode or prompt fields.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=repo_root / "examples/images",
        help="Directory containing manifest images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "outputs/baseline_fp32_manifest",
        help="Baseline output directory.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--expected-revision", default=FIXED_MODEL_REVISION)
    parser.add_argument("--expected-transformers-version", default=FIXED_TRANSFORMERS_VERSION)
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=524288)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_model_revision(model_dir: Path) -> str:
    metadata = model_dir / ".cache/huggingface/download/config.json.metadata"
    if not metadata.is_file():
        return "unknown"
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines and lines[0].strip() else "unknown"


def validate_model_revision(actual: str, expected: str) -> None:
    if actual in ("", "unknown", "<missing>"):
        fail("checkpoint revision metadata missing")
    if actual != expected:
        fail(f"checkpoint revision mismatch: got {actual}, expected {expected}")


def validate_transformers_version(actual: str, expected: str) -> None:
    if actual in ("", "unknown", "<missing>"):
        fail("Transformers version missing")
    if actual != expected:
        fail(f"Transformers version mismatch: got {actual}, expected {expected}")


def generation_options(max_new_tokens: int, repetition_penalty: float) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": repetition_penalty,
    }


def load_runtime_deps() -> None:
    global np
    global torch
    global transformers
    global Image
    global AutoProcessor
    global HunYuanVLForConditionalGeneration

    try:
        import accelerate  # noqa: F401 - required by Transformers device_map loading
        import numpy as numpy_module
        import torch as torch_module
        import transformers as transformers_module
        from PIL import Image as image_module
        from transformers import AutoProcessor as auto_processor_cls
        from transformers import HunYuanVLForConditionalGeneration as hunyuan_model_cls
    except Exception as exc:
        fail(
            "failed to import baseline dependencies; activate the HunyuanOCR Python environment "
            f"with torch/transformers/accelerate/pillow installed: {exc}"
        )

    np = numpy_module
    torch = torch_module
    transformers = transformers_module
    Image = image_module
    AutoProcessor = auto_processor_cls
    HunYuanVLForConditionalGeneration = hunyuan_model_cls


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"{label} not found: {path}")


def prompt_for_item(item: dict[str, Any]) -> tuple[str, str]:
    prompt_mode = item.get("prompt_mode")
    prompt = item.get("prompt")
    if (prompt_mode is None) == (prompt is None):
        fail(f"{item.get('name', '<unnamed>')}: manifest case must contain exactly one of prompt_mode or prompt")
    if prompt is not None:
        return str(prompt), "custom"
    if prompt_mode == "spotting":
        return SPOTTING_PROMPT, "spotting"
    if prompt_mode == "document":
        return DOCUMENT_PROMPT, "document"
    fail(f"{item.get('name', '<unnamed>')}: unsupported prompt_mode: {prompt_mode}")


def load_cases(manifest: Path) -> list[BaselineCase]:
    require_file(manifest, "manifest")
    items = json.loads(manifest.read_text(encoding="utf-8"))
    names = [str(item.get("name") or Path(item["image"]).stem) for item in items]
    try:
        validate_case_names(names)
    except ValueError as exc:
        fail(str(exc))
    cases: list[BaselineCase] = []
    for item, name in zip(items, names):
        prompt, label = prompt_for_item(item)
        cases.append(BaselineCase(name, str(item["image"]), prompt, label))
    return cases


def apply_processor_pixel_bounds(processor: AutoProcessor, min_pixels: int, max_pixels: int) -> dict[str, int]:
    image_processor = processor.image_processor
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    image_processor.size = {
        "shortest_edge": int(image_processor.min_pixels),
        "longest_edge": int(image_processor.max_pixels),
    }
    return {
        "min_pixels": int(image_processor.min_pixels),
        "max_pixels": int(image_processor.max_pixels),
    }


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


def normalize_chat_template_tokens(tokens: list[Any]) -> list[int]:
    if len(tokens) == 1 and isinstance(tokens[0], list):
        return tokens[0]
    return tokens


def to_numpy_int64(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.int64).numpy()


def to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def build_text_embeddings(model: HunYuanVLForConditionalGeneration, input_ids: Any) -> Any:
    return model.get_input_embeddings()(input_ids)


def encode_image_features(model: HunYuanVLForConditionalGeneration, pixel_values: Any, image_grid_thw: Any) -> Any:
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
    return model.vit(pixel_values, image_grid_thw)


def get_placeholder_mask(
    model: HunYuanVLForConditionalGeneration,
    input_ids: Any,
    inputs_embeds: Any,
    image_features: Any,
) -> Any:
    owner = model.model if hasattr(model.model, "get_placeholder_mask") else model
    return owner.get_placeholder_mask(
        input_ids,
        inputs_embeds=inputs_embeds,
        image_features=image_features,
    )


def build_fp32_inputs_embeds(
    model: HunYuanVLForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_embeds = build_text_embeddings(model, input_ids)
    vision_module = model.model.vision_tower if hasattr(model.model, "vision_tower") else model.vit
    vit_device = next(vision_module.parameters()).device
    image_embeds = encode_image_features(
        model,
        pixel_values.to(device=vit_device, dtype=torch.float32),
        image_grid_thw.to(device=vit_device),
    )
    image_embeds_for_llm = image_embeds.to(input_ids.device, non_blocking=True)
    image_mask = get_placeholder_mask(model, input_ids, inputs_embeds, image_embeds_for_llm)
    if isinstance(image_mask, tuple):
        image_mask = image_mask[0]
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_for_llm)
    return inputs_embeds, image_embeds


def resolve_position_ids(inputs: Any, model: HunYuanVLForConditionalGeneration) -> tuple[Any, Any]:
    position_ids = inputs.get("position_ids")
    if position_ids is not None:
        return position_ids, position_ids

    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        fail("processor returned neither position_ids nor mm_token_type_ids")
    position_ids = model.model.compute_3d_position_ids(
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
        mm_token_type_ids=mm_token_type_ids,
    )
    return position_ids, position_ids.permute(1, 0, 2)


def run_case(
    case: BaselineCase,
    processor: AutoProcessor,
    model: HunYuanVLForConditionalGeneration,
    image_root: Path,
    max_new_tokens: int,
    repetition_penalty: float,
    output_dir: Path,
) -> dict[str, Any]:
    image_path = image_root / case.image
    require_file(image_path, f"image for {case.name}")

    out_dir = output_dir / case.name
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    messages = build_messages(image_path, case.prompt)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_template_tokens = normalize_chat_template_tokens(
        processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    inputs = processor(text=[chat_text], images=image, padding=True, return_tensors="pt")

    image_token_id = int(model.config.image_token_id)
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]
    image_grid_thw_cpu = inputs["image_grid_thw"]
    image_token_count = int((input_ids_cpu == image_token_id).sum().item())
    model_position_ids, position_ids_cpu = resolve_position_ids(inputs, model)
    inputs["position_ids"] = model_position_ids

    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        inputs_embeds, image_embeds = build_fp32_inputs_embeds(
            model,
            inputs["input_ids"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
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
            **generation_options(max_new_tokens, repetition_penalty),
        )

    generated_ids_trimmed = generated_ids[:, input_ids_cpu.shape[1] :]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    save_npz(
        out_dir / "input_ids.npz",
        input_ids=to_numpy_int64(input_ids_cpu),
        attention_mask=to_numpy_int64(attention_mask_cpu),
        position_ids=to_numpy_int64(position_ids_cpu),
        image_grid_thw=to_numpy_int64(image_grid_thw_cpu),
        image_token_id=np.array([image_token_id], dtype=np.int64),
        image_token_count=np.array([image_token_count], dtype=np.int64),
    )
    (out_dir / "chat_template_tokens.txt").write_text(
        " ".join(str(int(x)) for x in chat_template_tokens) + "\n",
        encoding="utf-8",
    )
    save_npz(
        out_dir / "vision_features.npz",
        vision_features=to_numpy_float32(image_embeds),
        image_grid_thw=to_numpy_int64(image_grid_thw_cpu),
    )
    save_npz(out_dir / "first_logits.npz", logits=to_numpy_float32(first_outputs.logits))
    save_npz(
        out_dir / "generated_ids.npz",
        generated_ids=to_numpy_int64(generated_ids),
        generated_ids_trimmed=to_numpy_int64(generated_ids_trimmed),
    )
    (out_dir / "output_text.txt").write_text(output_text, encoding="utf-8")

    summary = {
        "name": case.name,
        "image": case.image,
        "prompt_label": case.prompt_label,
        "image_mode": image.mode,
        "image_size": list(image.size),
        "chat_template_token_len": len(chat_template_tokens),
        "input_ids_len": int(input_ids_cpu.shape[1]),
        "image_token_count": image_token_count,
        "image_grid_thw": image_grid_thw_cpu.tolist(),
        "vision_features_shape": list(image_embeds.shape),
        "first_logits_shape": list(first_outputs.logits.shape),
        "generated_ids_len": int(generated_ids.shape[1]),
        "new_tokens_len": int(generated_ids_trimmed.shape[1]),
        "output_text_chars": len(output_text),
        "output_text_preview": output_text[:160].replace("\n", "\\n"),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)

    del inputs, inputs_embeds, image_embeds, first_outputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def write_summary(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
    pixel_bounds: dict[str, int],
) -> None:
    lines = [
        "HunyuanOCR Transformers fp32 deterministic baseline",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"model_dir: {args.model_dir}",
        f"model_revision: {read_model_revision(args.model_dir)}",
        f"manifest: {args.manifest}",
        f"torch: {torch.__version__}",
        f"transformers: {transformers.__version__}",
        f"device: {args.device}",
        'attn_implementation: "eager"',
        "dtype: torch.float32",
        "image_embeds: manual fp32 vision tower",
        "generate: Transformers parent generate(inputs_embeds=...)",
        f"max_new_tokens: {args.max_new_tokens}",
        f"repetition_penalty: {args.repetition_penalty}",
        f"processor.min_pixels: {pixel_bounds['min_pixels']}",
        f"processor.max_pixels: {pixel_bounds['max_pixels']}",
        "do_sample: False",
        "",
    ]
    for item in summaries:
        lines.extend(
            [
                f"[{item['name']}]",
                f"image: {item['image']}",
                f"prompt_label: {item['prompt_label']}",
                f"image_grid_thw: {item['image_grid_thw']}",
                f"new_tokens_len: {item['new_tokens_len']}",
                f"output_text_chars: {item['output_text_chars']}",
                f"output_text_preview: {item['output_text_preview']}",
                "",
            ]
        )
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        fail("--max-new-tokens must be positive")
    if args.repetition_penalty <= 0:
        fail("--repetition-penalty must be positive")
    args.model_dir = args.model_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.image_root = args.image_root.resolve()
    args.output_dir = lexical_absolute_path(args.output_dir)

    cases = load_cases(args.manifest)
    validate_model_revision(read_model_revision(args.model_dir), args.expected_revision)
    if (
        paths_overlap(args.output_dir, args.model_dir)
        or paths_overlap(args.output_dir, args.image_root)
        or paths_overlap(args.output_dir, args.manifest)
    ):
        fail("output directory must not overlap model, image, or manifest paths")
    load_runtime_deps()
    validate_transformers_version(transformers.__version__, args.expected_transformers_version)
    print(f"Loading processor from {args.model_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(str(args.model_dir), use_fast=False)
    pixel_bounds = apply_processor_pixel_bounds(processor, args.min_pixels, args.max_pixels)
    print(f"Processor pixel bounds: {pixel_bounds}", flush=True)
    print(f"Loading fp32 model from {args.model_dir}", flush=True)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        str(args.model_dir),
        attn_implementation="eager",
        dtype=torch.float32,
        device_map=args.device,
    ).eval()

    def build(staging: Path) -> None:
        summaries = []
        for case in cases:
            print(f"Running {case.name} ({case.prompt_label})", flush=True)
            summaries.append(
                run_case(
                    case,
                    processor,
                    model,
                    args.image_root,
                    args.max_new_tokens,
                    args.repetition_penalty,
                    staging,
                )
            )
        write_summary(staging, summaries, args, pixel_bounds)

    build_directory_transactionally(args.output_dir, build, replace_existing=True)
    print(f"Wrote summary to {args.output_dir / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
