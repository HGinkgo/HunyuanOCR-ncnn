#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/linux-ci.yml").read_text(encoding="utf-8")
    required = (
        "actions/checkout@v7",
        "python3-numpy",
        "python scripts/apply_ncnn_patches.py --ncnn-dir _deps/ncnn",
        "ctest --test-dir build --output-on-failure",
    )
    missing = [value for value in required if value not in workflow]
    if missing:
        print(f"Linux workflow missing test dependencies/settings: {missing}", file=sys.stderr)
        return 1
    if workflow.count("actions/checkout@v7") != 2:
        print("Linux workflow must use Node.js 24 checkout for both repositories", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
