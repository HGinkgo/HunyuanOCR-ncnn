#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


NCNN_REVISION = "dda2e28bae2a084760361197d87f06e685604e52"
MATMUL_PR = "https://github.com/Tencent/ncnn/pull/6579"
MATMUL_COMMIT = "88e0927f6e6b640fea19bd5721ff5409fcca99ef"
PATCHES = (
    "0001-vulkan-matmul-pr6579.patch",
    "0002-vulkan-exact-gelu.patch",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def test_bundle_metadata(root: Path) -> None:
    bundle = root / "patches" / "ncnn"
    series = (bundle / "series").read_text(encoding="utf-8").splitlines()
    require(tuple(series) == PATCHES, "ncnn patch series must be ordered and complete")

    matmul = (bundle / PATCHES[0]).read_text(encoding="utf-8")
    require(NCNN_REVISION in matmul, "MatMul patch must record the pinned ncnn revision")
    require(MATMUL_PR in matmul, "MatMul patch must credit upstream PR #6579")
    require(MATMUL_COMMIT in matmul, "MatMul patch must record the source commit")
    require("Cat-myq" in matmul, "MatMul patch must credit Cat-myq")
    require("BSD 3-Clause" in matmul, "MatMul patch must identify the ncnn license")

    gelu = (bundle / PATCHES[1]).read_text(encoding="utf-8")
    require("Vulkan exact GELU" in gelu, "GELU patch must describe its purpose")
    require("BSD 3-Clause" in gelu, "GELU patch must identify the ncnn license")

    license_text = (bundle / "LICENSE.ncnn.txt").read_text(encoding="utf-8")
    for phrase in ("Redistribution and use", "Neither the name of [copyright holder]", "AS IS"):
        require(phrase in license_text, f"ncnn BSD license is incomplete: {phrase}")

    notice = (root / "NOTICE").read_text(encoding="utf-8")
    require("does not vendor ncnn source code" not in notice, "NOTICE contains stale no-vendoring claim")
    for value in (MATMUL_PR, MATMUL_COMMIT, "Cat-myq"):
        require(value in notice, f"NOTICE must include ncnn patch attribution: {value}")

    for name in ("README.md", "README_en.md"):
        readme = (root / name).read_text(encoding="utf-8")
        require(NCNN_REVISION in readme, f"{name} must report the pinned ncnn revision")
        require("patches/ncnn" in readme, f"{name} must explain the carried ncnn patches")


def test_wrong_revision_is_rejected(root: Path) -> None:
    script = root / "scripts" / "apply_ncnn_patches.py"
    require(script.is_file(), "missing cross-platform ncnn patch application script")

    with tempfile.TemporaryDirectory(prefix="hunyuan-ncnn-patch-test-") as temp:
        repo = Path(temp) / "ncnn"
        repo.mkdir()
        require(run("git", "init", "-q", cwd=repo).returncode == 0, "failed to init test repository")
        require(run("git", "config", "user.email", "test@example.com", cwd=repo).returncode == 0, "git config failed")
        require(run("git", "config", "user.name", "test", cwd=repo).returncode == 0, "git config failed")
        (repo / "README.md").write_text("test\n", encoding="utf-8")
        require(run("git", "add", "README.md", cwd=repo).returncode == 0, "git add failed")
        require(run("git", "commit", "-qm", "test", cwd=repo).returncode == 0, "git commit failed")

        result = subprocess.run(
            [sys.executable, str(script), "--ncnn-dir", str(repo), "--check"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        require(result.returncode != 0, "patch script must reject a non-pinned ncnn revision")
        require(NCNN_REVISION in result.stderr, "revision error must report the required ncnn revision")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    test_bundle_metadata(root)
    test_wrong_revision_is_rejected(root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
