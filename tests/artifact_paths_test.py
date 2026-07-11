#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from export import export_all
    from tools import package_model

    expected = Path("models/export/vision_dynamic")
    export_path = getattr(export_all, "DEFAULT_DYNAMIC_VISION_DIR", None)
    package_path = getattr(package_model, "DEFAULT_DYNAMIC_VISION_DIR", None)
    if export_path != expected:
        print(f"export_all dynamic vision path mismatch: {export_path}", file=sys.stderr)
        return 1
    if package_path != expected:
        print(f"package_model dynamic vision path mismatch: {package_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
