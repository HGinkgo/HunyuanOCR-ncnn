#!/usr/bin/env python3
"""Create C++ regression fixtures from HunyuanOCR fp32 baseline outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from atomic_output import build_directory_transactionally, lexical_absolute_path, paths_overlap, validate_case_names


FIXED_MODEL_REVISION = "9e01f897bf8956f77a80c350dc0491d6bbbd43e6"
FIXED_TRANSFORMERS_VERSION = "5.13.0"
REQUIRED_PROVENANCE = {
    "device": "cpu",
    "attn_implementation": '"eager"',
    "dtype": "torch.float32",
    "repetition_penalty": "1.08",
    "processor.min_pixels": "262144",
    "processor.max_pixels": "524288",
    "do_sample": "False",
}


@dataclass(frozen=True)
class Case:
    name: str
    image: str
    prompt_mode: str | None = None
    prompt: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    workspace = root.parent
    parser = argparse.ArgumentParser(description="Create VLM regression fixtures from baseline outputs.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=workspace / "outputs/baseline_fp32_p512k",
        help="Directory containing per-case fp32 baseline outputs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "examples/regression_cases.json",
        help="Regression case manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/regression_fixtures",
        help="Fixture output directory.",
    )
    parser.add_argument("--force", action="store_true", help="Remove output directory if it exists.")
    parser.add_argument("--expected-revision", default=FIXED_MODEL_REVISION)
    parser.add_argument("--expected-transformers-version", default=FIXED_TRANSFORMERS_VERSION)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def require_numpy() -> None:
    if np is None:
        fail("NumPy is required to convert regression fixtures; install it with `python -m pip install numpy`")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"{label} not found: {path}")


def read_baseline_provenance(summary_path: Path) -> dict[str, str]:
    if not summary_path.is_file():
        fail(f"baseline summary not found: {summary_path}")
    provenance: dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            break
        key, separator, value = line.partition(":")
        if separator:
            provenance[key.strip()] = value.strip()
    return provenance


def validate_baseline_provenance(
    baseline_dir: Path,
    expected_revision: str,
    expected_transformers_version: str,
) -> None:
    provenance = read_baseline_provenance(baseline_dir / "summary.txt")
    expected = {
        "model_revision": expected_revision,
        "transformers": expected_transformers_version,
        **REQUIRED_PROVENANCE,
    }
    for key, expected_value in expected.items():
        if key not in provenance:
            fail(f"baseline provenance {key} missing")
        actual = provenance[key]
        if actual in ("", "unknown", "<missing>"):
            fail(f"baseline provenance {key} missing")
        if actual != expected_value:
            fail(f"baseline provenance {key} mismatch: got {actual}, expected {expected_value}")


def load_cases(manifest: Path) -> list[Case]:
    require_file(manifest, "manifest")
    items = json.loads(manifest.read_text(encoding="utf-8"))
    try:
        validate_case_names([item.get("name") for item in items])
    except ValueError as exc:
        fail(str(exc))
    cases: list[Case] = []
    for item in items:
        prompt_mode = item.get("prompt_mode")
        prompt = item.get("prompt")
        if (prompt_mode is None) == (prompt is None):
            fail(f"{item.get('name', '<unnamed>')}: manifest case must contain exactly one of prompt_mode or prompt")
        cases.append(Case(item["name"], item["image"], prompt_mode, prompt))
    return cases


def write_i32(path: Path, values: np.ndarray) -> None:
    require_numpy()
    np.asarray(values, dtype=np.int32).reshape(-1).tofile(path)


def write_f32(path: Path, values: np.ndarray) -> None:
    require_numpy()
    np.asarray(values, dtype=np.float32).reshape(-1).tofile(path)


def prepare_case(baseline_dir: Path, output_dir: Path, case: Case) -> None:
    require_numpy()
    source = baseline_dir / case.name
    if not source.is_dir():
        fail(f"baseline case directory not found: {source}")

    input_npz = np.load(source / "input_ids.npz")
    generated_npz = np.load(source / "generated_ids.npz")
    vision_npz = np.load(source / "vision_features.npz")
    expected_text_path = source / "output_text.txt"
    require_file(expected_text_path, "expected text")

    input_ids = input_npz["input_ids"].reshape(-1)
    position_ids = input_npz["position_ids"].reshape(-1)
    image_token_id = int(input_npz["image_token_id"].reshape(-1)[0])
    vision_token_count = int(input_npz["image_token_count"].reshape(-1)[0])
    expected_tokens = generated_npz["generated_ids_trimmed"].reshape(-1)
    vision_features = vision_npz["vision_features"].reshape(-1)

    expected_feature_values = vision_token_count * 1024
    if vision_features.size != expected_feature_values:
        fail(
            f"{case.name}: vision feature count mismatch: "
            f"got {vision_features.size}, expected {expected_feature_values}"
        )

    target = output_dir / case.name
    target.mkdir(parents=True, exist_ok=True)
    write_i32(target / "input_ids.i32", input_ids)
    write_i32(target / "position_ids.i32", position_ids)
    write_i32(target / "expected_tokens.i32", expected_tokens)
    write_f32(target / "vision_features.f32", vision_features)
    shutil.copyfile(expected_text_path, target / "expected_text.txt")
    (target / "meta.txt").write_text(
        "\n".join(
            [
                f"seq_len={input_ids.size}",
                f"expected_token_count={expected_tokens.size}",
                f"image_token_id={image_token_id}",
                f"vision_token_count={vision_token_count}",
                f"case={case.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    baseline_dir = args.baseline_dir.resolve()
    manifest = args.manifest.resolve()
    output_dir = lexical_absolute_path(args.output_dir)

    validate_baseline_provenance(
        baseline_dir,
        args.expected_revision,
        args.expected_transformers_version,
    )
    cases = load_cases(manifest)
    if paths_overlap(output_dir, baseline_dir) or paths_overlap(output_dir, manifest):
        fail("output directory must not overlap baseline or manifest paths")
    if output_dir.exists() and not args.force:
        fail(f"output directory exists; pass --force: {output_dir}")

    def build(staging: Path) -> None:
        for case in cases:
            prepare_case(baseline_dir, staging, case)
            print(f"prepared {case.name}")

    build_directory_transactionally(output_dir, build, replace_existing=args.force)
    print(f"fixtures: {output_dir}")
    print(f"total: {len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
