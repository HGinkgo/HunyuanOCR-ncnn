#!/usr/bin/env python3
"""Apply the HunyuanOCR-ncnn Vulkan patch series to the pinned ncnn checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PINNED_NCNN_REVISION = "dda2e28bae2a084760361197d87f06e685604e52"


class PatchError(RuntimeError):
    pass


def run_git(ncnn_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ncnn_dir,
        text=True,
        capture_output=True,
        check=False,
    )


def require_pinned_revision(ncnn_dir: Path) -> None:
    result = run_git(ncnn_dir, "rev-parse", "HEAD")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PatchError(f"not a git checkout: {ncnn_dir}\n{detail}")

    actual = result.stdout.strip()
    if actual != PINNED_NCNN_REVISION:
        raise PatchError(
            "ncnn revision mismatch: "
            f"expected {PINNED_NCNN_REVISION}, got {actual}"
        )


def load_series(root: Path) -> list[Path]:
    bundle = root / "patches" / "ncnn"
    series_path = bundle / "series"
    try:
        names = [line.strip() for line in series_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise PatchError(f"cannot read patch series: {exc}") from exc

    if not names:
        raise PatchError("ncnn patch series is empty")

    patches: list[Path] = []
    for name in names:
        if Path(name).name != name:
            raise PatchError(f"invalid patch name in series: {name}")
        patch = bundle / name
        if not patch.is_file():
            raise PatchError(f"patch listed in series does not exist: {patch}")
        patches.append(patch)
    return patches


def check_patch(ncnn_dir: Path, patch: Path, reverse: bool = False) -> subprocess.CompletedProcess[str]:
    args = ["apply"]
    if reverse:
        args.append("--reverse")
    args.extend(["--check", str(patch)])
    return run_git(ncnn_dir, *args)


def apply_series(ncnn_dir: Path, patches: list[Path], check_only: bool) -> None:
    for patch in patches:
        if check_patch(ncnn_dir, patch, reverse=True).returncode == 0:
            print(f"already applied: {patch.name}")
            continue

        forward = check_patch(ncnn_dir, patch)
        if forward.returncode != 0:
            detail = forward.stderr.strip() or forward.stdout.strip()
            raise PatchError(f"cannot apply {patch.name}:\n{detail}")

        if check_only:
            print(f"applicable: {patch.name}")
            continue

        applied = run_git(ncnn_dir, "apply", str(patch))
        if applied.returncode != 0:
            detail = applied.stderr.strip() or applied.stdout.strip()
            raise PatchError(f"failed to apply {patch.name}:\n{detail}")
        print(f"applied: {patch.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the project-maintained Vulkan patches to the pinned ncnn "
            f"revision {PINNED_NCNN_REVISION}."
        )
    )
    parser.add_argument("--ncnn-dir", type=Path, required=True, help="path to the ncnn git checkout")
    parser.add_argument("--check", action="store_true", help="verify applicability without changing ncnn")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    ncnn_dir = args.ncnn_dir.expanduser().resolve()
    try:
        require_pinned_revision(ncnn_dir)
        apply_series(ncnn_dir, load_series(root), args.check)
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
