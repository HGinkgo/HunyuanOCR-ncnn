#!/usr/bin/env python3
"""Package exported HunyuanOCR ncnn artifacts into the runtime model layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


RUNTIME_FILES = [
    ("models/tokenizer/vocab.txt", "tokenizer/vocab.txt"),
    ("models/tokenizer/merges.txt", "tokenizer/merges.txt"),
    ("models/tokenizer/special_tokens.json", "tokenizer/special_tokens.json"),
    ("models/tokenizer/eos_ids.json", "tokenizer/eos_ids.json"),
    ("models/export/text_embed/text_embed.ncnn.param", "text_embed/text_embed.ncnn.param"),
    ("models/export/text_embed/text_embed.ncnn.bin", "text_embed/text_embed.ncnn.bin"),
    ("models/export/text_decoder/text_decoder_kv.ncnn.param", "text_decoder/text_decoder_kv.ncnn.param"),
    ("models/export/text_decoder/text_decoder_kv.ncnn.bin", "text_decoder/text_decoder_kv.ncnn.bin"),
    ("models/export/lm_head/lm_head.ncnn.param", "lm_head/lm_head.ncnn.param"),
    ("models/export/lm_head/lm_head.ncnn.bin", "lm_head/lm_head.ncnn.bin"),
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    default_summary = workspace_root / "models/export/vision/fp32_p512k/summary.json"
    default_dynamic_vision_dir = workspace_root / "models/export/vision_dynamic_probe/ncnn"

    parser = argparse.ArgumentParser(
        description="Create a standard HunyuanOCR-ncnn model directory from exported artifacts."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=workspace_root,
        help=f"Project workspace containing models/export and models/tokenizer. Default: {workspace_root}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination packaged model directory. It should stay outside git-tracked source.",
    )
    parser.add_argument(
        "--vision-summary",
        type=Path,
        default=default_summary,
        help=f"Vision export summary JSON. Default: {default_summary}",
    )
    parser.add_argument(
        "--vision-backend",
        choices=("fixed", "dynamic", "both"),
        default="fixed",
        help="Vision artifacts to package. Default keeps the v0.1 fixed-grid layout.",
    )
    parser.add_argument(
        "--dynamic-vision-dir",
        type=Path,
        default=default_dynamic_vision_dir,
        help=f"Directory containing vision_dynamic.ncnn.param/bin and pos_embed.bin. Default: {default_dynamic_vision_dir}",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy artifact files instead of creating symlinks. Useful on Windows or for portable bundles.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"required file not found: {path}")


def safe_prepare_output(output: Path, workspace: Path, repo_root: Path, force: bool) -> None:
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        workspace.resolve(),
        repo_root.resolve(),
        (workspace / "models").resolve(),
    }
    resolved = output.resolve()
    if resolved in protected:
        fail(f"refusing to use protected path as output: {resolved}")
    if output.exists() or output.is_symlink():
        if not force:
            fail(f"output already exists, pass --force to replace it: {output}")
        if output.is_symlink() or output.is_file():
            output.unlink()
        else:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)


def install_file(src: Path, dst: Path, copy_mode: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError as exc:
        fail(f"failed to create symlink {dst} -> {src}: {exc}; retry with --copy")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")


def unique_vision_grids(summary_path: Path) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        fail(f"vision summary has no cases: {summary_path}")

    grids: dict[tuple[int, int], dict[str, Any]] = {}
    for case in cases:
        grid_thw = case.get("image_grid_thw")
        if (
            not isinstance(grid_thw, list)
            or not grid_thw
            or not isinstance(grid_thw[0], list)
            or len(grid_thw[0]) != 3
        ):
            fail(f"invalid image_grid_thw in vision summary case: {case.get('case', '<unknown>')}")
        _, grid_h_raw, grid_w_raw = grid_thw[0]
        grid_h = int(grid_h_raw)
        grid_w = int(grid_w_raw)
        key = (grid_h, grid_w)
        if key in grids:
            continue

        param = Path(str(case.get("param", "")))
        bin_file = Path(str(case.get("bin", "")))
        require_file(param)
        require_file(bin_file)
        grids[key] = {
            "case": case.get("case", ""),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "param_src": param,
            "bin_src": bin_file,
            "param": f"vision/grid_{grid_h}x{grid_w}/vision.ncnn.param",
            "bin": f"vision/grid_{grid_h}x{grid_w}/vision.ncnn.bin",
        }

    return [grids[key] for key in sorted(grids)]


def load_manifest_template(repo_root: Path) -> dict[str, Any]:
    template_path = repo_root / "models/model.json.example"
    require_file(template_path)
    manifest = read_json(template_path)
    if not isinstance(manifest, dict):
        fail(f"manifest template is not a JSON object: {template_path}")
    return manifest


def write_manifest(repo_root: Path, output: Path, grids: list[dict[str, Any]]) -> None:
    manifest = load_manifest_template(repo_root)
    vision = manifest.setdefault("networks", {}).setdefault("vision", {})
    vision["status"] = "fixed_grid_vision_exports"
    vision["backend"] = "fixed_grid"
    vision["available_grids"] = [
        {
            "grid_h": grid["grid_h"],
            "grid_w": grid["grid_w"],
            "param": grid["param"],
            "bin": grid["bin"],
        }
        for grid in grids
    ]
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def dynamic_vision_files(dynamic_vision_dir: Path) -> dict[str, Path]:
    candidates = {
        "param": [
            dynamic_vision_dir / "vision.ncnn.param",
            dynamic_vision_dir / "vision_dynamic.ncnn.param",
        ],
        "bin": [
            dynamic_vision_dir / "vision.ncnn.bin",
            dynamic_vision_dir / "vision_dynamic.ncnn.bin",
        ],
        "pos_embed": [
            dynamic_vision_dir / "pos_embed.bin",
        ],
    }
    resolved: dict[str, Path] = {}
    for key, paths in candidates.items():
        for path in paths:
            if path.is_file():
                resolved[key] = path
                break
        if key not in resolved:
            fail(f"dynamic vision {key} file not found under: {dynamic_vision_dir}")
    return resolved


def install_dynamic_vision(dynamic_vision_dir: Path, output: Path, copy_mode: bool) -> int:
    files = dynamic_vision_files(dynamic_vision_dir)
    install_file(files["param"], output / "vision/vision.ncnn.param", copy_mode)
    install_file(files["bin"], output / "vision/vision.ncnn.bin", copy_mode)
    install_file(files["pos_embed"], output / "vision/pos_embed.bin", copy_mode)
    return 3


def write_dynamic_manifest(repo_root: Path, output: Path, include_fixed: bool, grids: list[dict[str, Any]]) -> None:
    manifest = load_manifest_template(repo_root)
    vision = manifest.setdefault("networks", {}).setdefault("vision", {})
    vision["status"] = "dynamic_vision" if not include_fixed else "dynamic_vision_with_fixed_grid_fallback"
    vision["backend"] = "dynamic"
    vision["param"] = "vision/vision.ncnn.param"
    vision["bin"] = "vision/vision.ncnn.bin"
    vision["pos_embed"] = "vision/pos_embed.bin"
    vision["pos_embed_shape"] = [1152, 128, 128]
    vision["pos_embed_interpolation"] = {
        "mode": "bilinear",
        "align_corners": False,
        "scale_factor": "((grid_h + 0.1) / 128, (grid_w + 0.1) / 128)",
    }
    if include_fixed:
        vision["fallback_backend"] = "fixed_grid"
        vision["available_grids"] = [
            {
                "grid_h": grid["grid_h"],
                "grid_w": grid["grid_w"],
                "param": grid["param"],
                "bin": grid["bin"],
            }
            for grid in grids
        ]
    else:
        vision["available_grids"] = []
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    vision_summary = args.vision_summary.resolve()
    dynamic_vision_dir = args.dynamic_vision_dir.resolve()

    if args.vision_backend in ("fixed", "both"):
        require_file(vision_summary)
    for src_rel, _ in RUNTIME_FILES:
        require_file(workspace / src_rel)

    grids = unique_vision_grids(vision_summary) if args.vision_backend in ("fixed", "both") else []
    if args.vision_backend in ("dynamic", "both"):
        dynamic_vision_files(dynamic_vision_dir)
    safe_prepare_output(output, workspace, repo_root, args.force)

    copied_files = 0
    for src_rel, dst_rel in RUNTIME_FILES:
        install_file(workspace / src_rel, output / dst_rel, args.copy)
        copied_files += 1

    if args.vision_backend in ("fixed", "both"):
        for grid in grids:
            install_file(grid["param_src"], output / grid["param"], args.copy)
            install_file(grid["bin_src"], output / grid["bin"], args.copy)
            copied_files += 2

    if args.vision_backend in ("dynamic", "both"):
        copied_files += install_dynamic_vision(dynamic_vision_dir, output, args.copy)
        write_dynamic_manifest(repo_root, output, args.vision_backend == "both", grids)
    else:
        write_manifest(repo_root, output, grids)
    copied_files += 1

    mode = "copy" if args.copy else "symlink"
    grid_names = ", ".join(f"{grid['grid_h']}x{grid['grid_w']}" for grid in grids) if grids else "(none)"
    print(f"packaged_model: {output}")
    print(f"mode: {mode}")
    print(f"vision_backend: {args.vision_backend}")
    print(f"runtime_files: {copied_files}")
    print(f"vision_grids: {grid_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
