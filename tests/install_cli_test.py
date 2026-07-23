#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: install_cli_test.py BUILD_DIR EXECUTABLE_SUFFIX", file=sys.stderr)
        return 2

    build_dir = Path(sys.argv[1]).resolve()
    suffix = sys.argv[2]
    with tempfile.TemporaryDirectory(prefix="hunyuan_install_") as temporary:
        prefix = Path(temporary)
        completed = subprocess.run(
            ["cmake", "--install", str(build_dir), "--prefix", str(prefix)],
            text=True,
            capture_output=True,
            check=False,
        )
        installed = prefix / "bin" / f"hunyuan-ocr{suffix}"
        if completed.returncode != 0 or not installed.is_file():
            print(
                "installed hunyuan-ocr command was not produced: "
                f"rc={completed.returncode} stdout={completed.stdout!r} "
                f"stderr={completed.stderr!r} expected={installed}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
