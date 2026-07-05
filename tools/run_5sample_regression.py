#!/usr/bin/env python3
"""Run the five-sample HunyuanOCR-ncnn fixture regression."""

from __future__ import annotations

import argparse
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
    prompt_mode: str


CASES = [
    Case("hf_demo_tools-dark", "hf_demo_tools-dark.png", "spotting"),
    Case(
        "omnidoc_document_zh_page-205e4273-5b94-43e5-bfaf-dc882416b067",
        "omnidoc_document_zh_page-205e4273-5b94-43e5-bfaf-dc882416b067.png",
        "spotting",
    ),
    Case(
        "omnidoc_document_book_docstructbench_enbook_19221575_1173",
        "omnidoc_document_book_docstructbench_enbook_19221575_1173.jpg",
        "document",
    ),
    Case(
        "omnidoc_formula_harmonic_analysis_page_119",
        "omnidoc_formula_harmonic_analysis_page_119.png",
        "document",
    ),
    Case(
        "omnidoc_table_pyomo_page_188",
        "omnidoc_table_pyomo_page_188.png",
        "document",
    ),
]

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

    parser = argparse.ArgumentParser(description="Run the fixed five-sample HunyuanOCR-ncnn regression.")
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
        default=workspace_root / "datasets/test_images",
        help="Directory containing the five test images.",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=temp_root / "hunyuanocr_phase53_vlm_fixtures",
        help="Directory containing VLM oracle fixtures.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=temp_root / "hunyuanocr_5sample_regression",
        help="Directory for per-case CLI logs.",
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
        "--prompt-mode",
        case.prompt_mode,
        "--vlm-fixture",
        str(fixture),
    ]

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
    print(f"{status} {case.name} ({case.prompt_mode})")
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

    require_file(args.binary, "hunyuan_ocr_cli")
    if args.package:
        run_packager(repo_root, args)
    require_dir(args.model, "packaged model")
    require_dir(args.image_root, "image root")
    require_dir(args.fixture_root, "fixture root")

    if args.log_dir.exists():
        shutil.rmtree(args.log_dir)
    args.log_dir.mkdir(parents=True)

    passed = 0
    for case in CASES:
        if run_case(repo_root, args, case):
            passed += 1

    print(f"summary: {passed}/{len(CASES)} passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
