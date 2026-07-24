#!/usr/bin/env python3
"""Package exported HunyuanOCR ncnn artifacts into the runtime model layout."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_DYNAMIC_VISION_DIR = Path("models/export/vision_dynamic")
DEFAULT_DFLASH_DIR = Path("models/export/dflash/ncnn")
DEFAULT_DFLASH_DECODER_DIR = Path("models/export/text_decoder_dflash_aux")

BASE_DECODER_DST = {
    "text_decoder/text_decoder_kv.ncnn.param",
    "text_decoder/text_decoder_kv.ncnn.bin",
}

STOCK_RUNTIME_FILES = [
    ("models/tokenizer/vocab.txt", "tokenizer/vocab.txt"),
    ("models/tokenizer/merges.txt", "tokenizer/merges.txt"),
    ("models/tokenizer/special_tokens.json", "tokenizer/special_tokens.json"),
    ("models/tokenizer/eos_ids.json", "tokenizer/eos_ids.json"),
    ("models/export/text_embed/text_embed.ncnn.param", "text_embed/text_embed.ncnn.param"),
    ("models/export/text_embed/text_embed.ncnn.bin", "text_embed/text_embed.ncnn.bin"),
    ("models/export/text_decoder/text_decoder_kv.ncnn.param", "text_decoder/text_decoder_kv.ncnn.param"),
    ("models/export/text_decoder/text_decoder_kv.ncnn.bin", "text_decoder/text_decoder_kv.ncnn.bin"),
    ("models/export/lm_head/lm_head.ncnn.param", "lm_head/lm_head.ncnn.param"),
]

# Kept for import compatibility with tests/artifact_paths_test.py helpers and callers.
RUNTIME_FILES = STOCK_RUNTIME_FILES
BASE_DECODER_FILES = [
    src_dst for src_dst in STOCK_RUNTIME_FILES if src_dst[1] in BASE_DECODER_DST
]
RUNTIME_FILES_WITHOUT_DECODER = [
    src_dst for src_dst in STOCK_RUNTIME_FILES if src_dst[1] not in BASE_DECODER_DST
]

BASE_RUNTIME_FILES = [
    "tokenizer/vocab.txt",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens.json",
    "tokenizer/eos_ids.json",
    "text_embed/text_embed.ncnn.param",
    "text_embed/text_embed.ncnn.bin",
    "text_decoder/text_decoder_kv.ncnn.param",
    "text_decoder/text_decoder_kv.ncnn.bin",
    "lm_head/lm_head.ncnn.param",
]

BASE_RUNTIME_FILES_WITHOUT_DECODER = [
    rel for rel in BASE_RUNTIME_FILES if rel not in BASE_DECODER_DST
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent

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
        "--dynamic-vision-dir",
        type=Path,
        default=None,
        help="Directory containing vision.ncnn.param/bin and pos_embed.bin. Default: <workspace>/models/export/vision_dynamic/ncnn",
    )
    parser.add_argument(
        "--base-runtime-dir",
        type=Path,
        default=None,
        help=(
            "Existing packaged runtime directory providing tokenizer/text_embed/text_decoder/lm_head. "
            "When set, stock export layout paths are not required for those files."
        ),
    )
    parser.add_argument(
        "--dflash",
        action="store_true",
        help="Package DFlash draft weights and the auxiliary text decoder used by DFlash.",
    )
    parser.add_argument(
        "--dflash-dir",
        type=Path,
        default=None,
        help="Directory containing dflash.ncnn.param/bin. Default: <workspace>/models/export/dflash/ncnn",
    )
    parser.add_argument(
        "--dflash-decoder-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the auxiliary text_decoder_kv.ncnn.param/bin used with DFlash. "
            "Default: <workspace>/models/export/text_decoder_dflash_aux"
        ),
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


def require_tied_weights_match(text_embed_bin: Path, lm_head_bin: Path) -> None:
    require_file(text_embed_bin)
    require_file(lm_head_bin)
    if not filecmp.cmp(text_embed_bin, lm_head_bin, shallow=False):
        fail(
            "tied text embedding and lm_head weights differ: "
            f"{text_embed_bin} != {lm_head_bin}"
        )


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


def load_manifest_template(repo_root: Path) -> dict[str, Any]:
    template_path = repo_root / "models/model.json.example"
    require_file(template_path)
    manifest = read_json(template_path)
    if not isinstance(manifest, dict):
        fail(f"manifest template is not a JSON object: {template_path}")
    return manifest


def dynamic_vision_files(dynamic_vision_dir: Path) -> dict[str, Path]:
    paths = {
        "param": dynamic_vision_dir / "vision.ncnn.param",
        "bin": dynamic_vision_dir / "vision.ncnn.bin",
        "pos_embed": dynamic_vision_dir / "pos_embed.bin",
    }
    for key, path in paths.items():
        if not path.is_file():
            fail(f"dynamic vision {key} file not found under: {dynamic_vision_dir}")
    return paths


def install_dynamic_vision(dynamic_vision_dir: Path, output: Path, copy_mode: bool) -> int:
    files = dynamic_vision_files(dynamic_vision_dir)
    install_file(files["param"], output / "vision/vision.ncnn.param", copy_mode)
    install_file(files["bin"], output / "vision/vision.ncnn.bin", copy_mode)
    install_file(files["pos_embed"], output / "vision/pos_embed.bin", copy_mode)
    return 3


def write_dynamic_manifest(repo_root: Path, output: Path) -> None:
    manifest = load_manifest_template(repo_root)
    vision = manifest.setdefault("networks", {}).setdefault("vision", {})
    vision["status"] = "dynamic_vision"
    vision["backend"] = "dynamic"
    vision["param"] = "vision/vision.ncnn.param"
    vision["bin"] = "vision/vision.ncnn.bin"
    vision["pos_embed"] = "vision/pos_embed.bin"
    vision["pos_embed_shape"] = [1152, 128, 128]
    vision["pos_embed_interpolation"] = {
        "mode": "bilinear",
        "align_corners": False,
        "size": ["grid_h", "grid_w"],
    }
    manifest["networks"]["lm_head"]["bin"] = "text_embed/text_embed.ncnn.bin"
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    dynamic_vision_dir = (
        args.dynamic_vision_dir.resolve()
        if args.dynamic_vision_dir is not None
        else workspace / DEFAULT_DYNAMIC_VISION_DIR / "ncnn"
    )
    dflash_dir = (
        args.dflash_dir.resolve()
        if args.dflash_dir is not None
        else workspace / DEFAULT_DFLASH_DIR
    )
    dflash_decoder_dir = (
        args.dflash_decoder_dir.resolve()
        if args.dflash_decoder_dir is not None
        else workspace / DEFAULT_DFLASH_DECODER_DIR
    )
    base_runtime_dir = (
        args.base_runtime_dir.resolve() if args.base_runtime_dir is not None else None
    )

    if (args.dflash_dir is not None or args.dflash_decoder_dir is not None) and not args.dflash:
        fail("--dflash-dir and --dflash-decoder-dir require --dflash")

    # List of (src_path, dst_rel) for non-vision core runtime files.
    core_sources: list[tuple[Path, str]] = []
    if base_runtime_dir is not None:
        base_embed_bin = base_runtime_dir / "text_embed/text_embed.ncnn.bin"
        legacy_lm_head_bin = base_runtime_dir / "lm_head/lm_head.ncnn.bin"
        require_file(base_embed_bin)
        if legacy_lm_head_bin.is_file():
            require_tied_weights_match(base_embed_bin, legacy_lm_head_bin)
        runtime_rels = (
            BASE_RUNTIME_FILES_WITHOUT_DECODER if args.dflash else BASE_RUNTIME_FILES
        )
        for rel in runtime_rels:
            src = base_runtime_dir / rel
            require_file(src)
            core_sources.append((src, rel))
    else:
        require_tied_weights_match(
            workspace / "models/export/text_embed/text_embed.ncnn.bin",
            workspace / "models/export/lm_head/lm_head.ncnn.bin",
        )
        stock_files = RUNTIME_FILES_WITHOUT_DECODER if args.dflash else STOCK_RUNTIME_FILES
        for src_rel, dst_rel in stock_files:
            src = workspace / src_rel
            require_file(src)
            core_sources.append((src, dst_rel))

    if args.dflash:
        aux_decoder_param = dflash_decoder_dir / "text_decoder_kv.ncnn.param"
        aux_decoder_bin = dflash_decoder_dir / "text_decoder_kv.ncnn.bin"
        dflash_param = dflash_dir / "dflash.ncnn.param"
        dflash_bin = dflash_dir / "dflash.ncnn.bin"
        for path in (aux_decoder_param, aux_decoder_bin, dflash_param, dflash_bin):
            require_file(path)
        core_sources.extend(
            [
                (aux_decoder_param, "text_decoder/text_decoder_kv.ncnn.param"),
                (aux_decoder_bin, "text_decoder/text_decoder_kv.ncnn.bin"),
                (dflash_param, "dflash/dflash.ncnn.param"),
                (dflash_bin, "dflash/dflash.ncnn.bin"),
            ]
        )

    dynamic_vision_files(dynamic_vision_dir)
    safe_prepare_output(output, workspace, repo_root, args.force)

    copied_files = 0
    for src_path, dst_rel in core_sources:
        install_file(src_path, output / dst_rel, args.copy)
        copied_files += 1

    copied_files += install_dynamic_vision(dynamic_vision_dir, output, args.copy)
    write_dynamic_manifest(repo_root, output)
    copied_files += 1

    mode = "copy" if args.copy else "symlink"
    print(f"packaged_model: {output}")
    print(f"mode: {mode}")
    print("vision_backend: dynamic")
    print("tied_text_weights: shared")
    print(f"runtime_files: {copied_files}")
    print(f"dflash: {'enabled' if args.dflash else 'disabled'}")
    print(
        "base_runtime_dir: "
        + (str(base_runtime_dir) if base_runtime_dir is not None else "stock-export")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
