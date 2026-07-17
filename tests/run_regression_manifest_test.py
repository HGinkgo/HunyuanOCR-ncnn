#!/usr/bin/env python3
"""Regression runner manifest behavior tests."""

from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def write_fake_binary(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

Path = __import__("pathlib").Path
Path(os.environ["ARGV_LOG"]).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
print("  image_grid_thw: [1, 1, 1]")
print("  generated_token_count: 1")
print("  match_fixture_input_ids: true")
print("  match_fixture_position_ids: true")
print("  match_expected_tokens: true")
print("  match_expected_text: true")
print("vision_gelu_cpu_fallback_count: 28")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def verify_canonical_regression_image(repo_root: Path) -> bool:
    manifest_path = repo_root / "examples/regression_cases.json"
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_images = {
        "hunyuan_vis_art_16": "hunyuan_vis_art_16.png",
        "hunyuan_ie_parallel": "hunyuan_ie_parallel.png",
    }
    for name, expected_image in canonical_images.items():
        case = next((item for item in items if item.get("name") == name), None)
        if case is None:
            print(f"{name} missing from regression manifest", file=sys.stderr)
            return False
        if case.get("image") != expected_image:
            print(f"{name} must use canonical PNG, got: {case.get('image')}", file=sys.stderr)
            return False
        image_path = repo_root / "examples/images" / expected_image
        if not image_path.is_file():
            print(f"canonical PNG missing: {image_path}", file=sys.stderr)
            return False
    return True


def verify_packaged_dynamic_manifest(repo_root: Path, output: Path) -> bool:
    spec = importlib.util.spec_from_file_location("package_model", repo_root / "tools/package_model.py")
    if spec is None or spec.loader is None:
        print("failed to load package_model.py", file=sys.stderr)
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_dynamic_manifest(repo_root, output)
    manifest = json.loads((output / "model.json").read_text(encoding="utf-8"))
    interpolation = manifest["networks"]["vision"]["pos_embed_interpolation"]
    if interpolation.get("size") != ["grid_h", "grid_w"] or "scale_factor" in interpolation:
        print("packaged model must use Transformers 5.13 size interpolation", file=sys.stderr)
        return False
    return True


def verify_hunyuan_1_5_model_defaults(repo_root: Path) -> bool:
    manifest = json.loads((repo_root / "models/model.json.example").read_text(encoding="utf-8"))
    baseline = manifest.get("baseline", {})
    if baseline.get("repetition_penalty") != 1.08:
        print(f"model default repetition penalty must be 1.08, got: {baseline.get('repetition_penalty')}", file=sys.stderr)
        return False
    if baseline.get("eos_ids") != [120020]:
        print(f"model default EOS ids must be [120020], got: {baseline.get('eos_ids')}", file=sys.stderr)
        return False
    vision = manifest.get("networks", {}).get("vision", {})
    stale_fixed_keys = {"fallback_backend", "directory_pattern", "param_name", "bin_name", "available_grids"}
    if vision.get("backend") != "dynamic" or stale_fixed_keys.intersection(vision):
        print("HunyuanOCR 1.5 model template must use only canonical dynamic vision", file=sys.stderr)
        return False
    interpolation = vision.get("pos_embed_interpolation", {})
    if interpolation.get("size") != ["grid_h", "grid_w"] or "scale_factor" in interpolation:
        print("HunyuanOCR 1.5 model template must use Transformers 5.13 size interpolation", file=sys.stderr)
        return False
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if not verify_canonical_regression_image(repo_root):
        return 1
    if not verify_hunyuan_1_5_model_defaults(repo_root):
        return 1
    with tempfile.TemporaryDirectory(prefix="hunyuanocr_run_regression_test_") as tmp_text:
        tmp = Path(tmp_text)
        model = tmp / "model"
        images = tmp / "images"
        fixtures = tmp / "fixtures"
        logs = tmp / "logs"
        fixture_case = fixtures / "custom_prompt_case"
        model.mkdir()
        images.mkdir()
        fixture_case.mkdir(parents=True)
        if not verify_packaged_dynamic_manifest(repo_root, tmp):
            return 1
        (images / "image.png").write_bytes(b"not a real image; fake binary does not read it")

        manifest = tmp / "manifest.json"
        prompt = "只输出图片中的可见文字"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "name": "custom_prompt_case",
                        "image": "image.png",
                        "prompt": prompt,
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        fake_binary = tmp / "fake_hunyuan_ocr_cli.py"
        argv_log = tmp / "argv.json"
        write_fake_binary(fake_binary)

        env = os.environ.copy()
        env["ARGV_LOG"] = str(argv_log)
        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "tools/run_regression.py"),
                "--binary",
                str(fake_binary),
                "--model",
                str(model),
                "--image-root",
                str(images),
                "--fixture-root",
                str(fixtures),
                "--manifest",
                str(manifest),
                "--log-dir",
                str(logs),
                "--vision-vulkan",
                "--vision-vulkan-device",
                "2",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            env=env,
        )
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode
        if "vision_gelu_cpu_fallback_count: 28" not in completed.stdout:
            print("regression summary omitted GELU CPU fallback count", file=sys.stderr)
            return 1

        argv = json.loads(argv_log.read_text(encoding="utf-8"))
        if "--prompt" not in argv:
            print(f"missing --prompt in argv: {argv}", file=sys.stderr)
            return 1
        if prompt not in argv:
            print(f"missing prompt text in argv: {argv}", file=sys.stderr)
            return 1
        if "--prompt-mode" in argv:
            print(f"unexpected --prompt-mode in argv: {argv}", file=sys.stderr)
            return 1
        if "--vision-vulkan" not in argv:
            print(f"missing --vision-vulkan in argv: {argv}", file=sys.stderr)
            return 1
        device_index = argv.index("--vision-vulkan-device") if "--vision-vulkan-device" in argv else -1
        if device_index < 0 or argv[device_index + 1] != "2":
            print(f"wrong Vulkan device in argv: {argv}", file=sys.stderr)
            return 1
        penalty_index = argv.index("--repetition-penalty") if "--repetition-penalty" in argv else -1
        if penalty_index < 0 or argv[penalty_index + 1] != "1.08":
            print(f"missing repetition penalty in argv: {argv}", file=sys.stderr)
            return 1
        if "summary: 1/1 passed" not in completed.stdout:
            print(completed.stdout, end="")
            print("missing expected summary", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
