#!/usr/bin/env python3
"""Run the HunyuanOCR-ncnn fixture regression cases."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    image: str
    prompt_mode: str | None = None
    prompt: str | None = None

    @property
    def label(self) -> str:
        return self.prompt_mode if self.prompt_mode is not None else "custom"


REQUIRED_MARKERS = [
    "match_fixture_input_ids: true",
    "match_fixture_position_ids: true",
    "match_expected_tokens: true",
    "match_expected_text: true",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    temp_root = Path(tempfile.gettempdir())

    parser = argparse.ArgumentParser(description="Run the HunyuanOCR-ncnn fixture regression.")
    parser.add_argument(
        "--binary",
        type=Path,
        default=repo_root / "build/hunyuan_ocr_cli",
        help="Path to hunyuan_ocr_cli.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=temp_root / "hunyuanocr_ncnn_model_packaged",
        help="Packaged model directory.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=repo_root / "examples/images",
        help="Directory containing the bundled test images.",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=temp_root / "hunyuanocr_regression_fixtures",
        help="Directory containing VLM oracle fixtures.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=temp_root / "hunyuanocr_regression",
        help="Directory for per-case CLI logs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "examples/regression_cases.json",
        help="Regression case manifest JSON.",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Run tools/package_model.py before regression.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=workspace_root,
        help="Workspace passed to package_model.py when --package is set.",
    )
    parser.add_argument(
        "--copy-package",
        action="store_true",
        help="Pass --copy to package_model.py when --package is set.",
    )
    parser.add_argument(
        "--package-vision-backend",
        choices=("fixed", "dynamic", "both"),
        default="fixed",
        help="Vision backend passed to package_model.py when --package is set.",
    )
    parser.add_argument(
        "--dynamic-vision-dir",
        type=Path,
        default=None,
        help="Dynamic vision artifact directory passed to package_model.py when --package is set.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
        help="Repetition penalty passed to hunyuan_ocr_cli. Default: 1.08.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        fail(f"{label} not found: {path}")


def load_cases(manifest: Path) -> list[Case]:
    require_file(manifest, "regression manifest")
    items = json.loads(manifest.read_text(encoding="utf-8"))
    cases: list[Case] = []
    for item in items:
        prompt_mode = item.get("prompt_mode")
        prompt = item.get("prompt")
        if (prompt_mode is None) == (prompt is None):
            fail(f"{item.get('name', '<unnamed>')}: manifest case must contain exactly one of prompt_mode or prompt")
        cases.append(Case(item["name"], item["image"], prompt_mode, prompt))
    return cases


def run_packager(repo_root: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "tools/package_model.py"),
        "--workspace",
        str(args.workspace),
        "--output",
        str(args.model),
        "--force",
    ]
    if args.copy_package:
        cmd.append("--copy")
    if args.package_vision_backend != "fixed":
        cmd.extend(["--vision-backend", args.package_vision_backend])
    if args.dynamic_vision_dir is not None:
        cmd.extend(["--dynamic-vision-dir", str(args.dynamic_vision_dir)])
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        fail(f"package_model.py failed with exit code {completed.returncode}")


def relevant_lines(output: str) -> list[str]:
    prefixes = (
        "  image_grid_thw:",
        "  generated_token_count:",
        "match_fixture_input_ids:",
        "match_fixture_position_ids:",
        "match_expected_tokens:",
        "match_expected_text:",
    )
    return [line for line in output.splitlines() if line.startswith(prefixes)]


def run_case(repo_root: Path, args: argparse.Namespace, case: Case) -> bool:
    image = args.image_root / case.image
    fixture = args.fixture_root / case.name
    require_file(image, f"image for {case.name}")
    require_dir(fixture, f"fixture for {case.name}")

    log_path = args.log_dir / f"{case.name}.log"
    cmd = [
        str(args.binary),
        "--model",
        str(args.model),
        "--image",
        str(image),
        "--repetition-penalty",
        str(args.repetition_penalty),
    ]
    if case.prompt is not None:
        cmd.extend(["--prompt", case.prompt])
    else:
        cmd.extend(["--prompt-mode", case.prompt_mode or ""])
    cmd.extend(["--vlm-fixture", str(fixture)])

    completed = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\n"
        + "## stdout\n"
        + completed.stdout
        + "\n## stderr\n"
        + completed.stderr,
        encoding="utf-8",
    )

    combined = completed.stdout + completed.stderr
    missing = [marker for marker in REQUIRED_MARKERS if marker not in combined]
    ok = completed.returncode == 0 and not missing

    status = "PASS" if ok else "FAIL"
    print(f"{status} {case.name} ({case.label})")
    for line in relevant_lines(combined):
        print(f"  {line.strip()}")
    if missing:
        print(f"  missing markers: {', '.join(missing)}")
    if completed.returncode != 0:
        print(f"  exit code: {completed.returncode}")
    print(f"  log: {log_path}")
    return ok


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args()
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.image_root = args.image_root.resolve()
    args.fixture_root = args.fixture_root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.workspace = args.workspace.resolve()
    args.manifest = args.manifest.resolve()
    if args.repetition_penalty <= 0:
        fail("--repetition-penalty must be positive")

    require_file(args.binary, "hunyuan_ocr_cli")
    if args.package:
        run_packager(repo_root, args)
    require_dir(args.model, "packaged model")
    require_dir(args.image_root, "image root")
    require_dir(args.fixture_root, "fixture root")
    cases = load_cases(args.manifest)

    if args.log_dir.exists():
        shutil.rmtree(args.log_dir)
    args.log_dir.mkdir(parents=True)

    passed = 0
    for case in cases:
        if run_case(repo_root, args, case):
            passed += 1

    print(f"summary: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
