#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "scripts/quickstart_existing_model.sh"

    with tempfile.TemporaryDirectory(prefix="hunyuan_quickstart_") as temporary:
        temporary_root = Path(temporary)
        repo = temporary_root / "repo"
        scripts = repo / "scripts"
        scripts.mkdir(parents=True)
        quickstart = scripts / source.name
        shutil.copyfile(source, quickstart)
        quickstart.chmod(0o755)

        model = repo / "hunyuan_ocr_ncnn_model"
        model.mkdir()
        ncnn_dir = temporary_root / "ncnn/build/src"
        ncnn_dir.mkdir(parents=True)
        (ncnn_dir / "ncnnConfig.cmake").touch()

        fake_bin = temporary_root / "bin"
        fake_bin.mkdir()
        cmake_log = temporary_root / "cmake.log"
        smoke_log = temporary_root / "smoke.log"
        write_executable(
            fake_bin / "cmake",
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CMAKE_LOG"\n',
        )
        write_executable(
            scripts / "smoke_test.sh",
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$SMOKE_LOG"\n',
        )

        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
        env["CMAKE_LOG"] = str(cmake_log)
        env["SMOKE_LOG"] = str(smoke_log)

        detected = subprocess.run(
            [str(quickstart)],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        cmake_args = cmake_log.read_text(encoding="utf-8") if cmake_log.exists() else ""
        smoke_args = smoke_log.read_text(encoding="utf-8") if smoke_log.exists() else ""
        if (
            detected.returncode != 0
            or f"Using model: {model}" not in detected.stdout
            or f"Using ncnn package: {ncnn_dir}" not in detected.stdout
            or f"-Dncnn_DIR={ncnn_dir}" not in cmake_args
            or f"--model {model}" not in smoke_args
            or "--max-tokens 16" not in smoke_args
        ):
            print(
                "quickstart did not auto-detect the conventional layout: "
                f"rc={detected.returncode} stdout={detected.stdout!r} "
                f"stderr={detected.stderr!r} cmake={cmake_args!r} smoke={smoke_args!r}",
                file=sys.stderr,
            )
            return 1

        invalid_ncnn = subprocess.run(
            [
                str(quickstart),
                "--model",
                str(model),
                "--ncnn-dir",
                str(temporary_root / "missing-ncnn"),
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            invalid_ncnn.returncode == 0
            or "ncnnConfig.cmake was not found" not in invalid_ncnn.stderr
        ):
            print(
                "invalid explicit ncnn path did not fail clearly: "
                f"rc={invalid_ncnn.returncode} stderr={invalid_ncnn.stderr!r}",
                file=sys.stderr,
            )
            return 1

        invalid_model = subprocess.run(
            [
                str(quickstart),
                "--model",
                str(temporary_root / "missing-model"),
                "--ncnn-dir",
                str(ncnn_dir),
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            invalid_model.returncode == 0
            or "model directory was not found" not in invalid_model.stderr
        ):
            print(
                "invalid explicit model path did not fail clearly: "
                f"rc={invalid_model.returncode} stderr={invalid_model.stderr!r}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
