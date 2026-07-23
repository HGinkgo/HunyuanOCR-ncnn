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
    if (root / "scripts/smoke_test.sh").exists():
        print("obsolete 16-token smoke_test.sh is still present", file=sys.stderr)
        return 1
    for readme_name in ("README.md", "README_en.md"):
        readme = (root / readme_name).read_text(encoding="utf-8")
        forbidden = (
            "smoke_test.sh",
            "16 个 token",
            "16-token",
            "--vision-vulkan",
            "--text-vulkan",
        )
        if any(marker in readme for marker in forbidden):
            print(f"{readme_name} retains an obsolete product entry", file=sys.stderr)
            return 1
        if "hunyuan-ocr --model ./hunyuan_ocr_ncnn_model --interactive" not in readme:
            print(
                f"{readme_name} does not show the installed interactive command with its model",
                file=sys.stderr,
            )
            return 1

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
        ncnn_root = temporary_root / "ncnn"
        ncnn_dir = ncnn_root / "build/src"
        ncnn_dir.mkdir(parents=True)
        (ncnn_dir / "ncnnConfig.cmake").touch()
        (ncnn_dir / "libncnn.a").touch()
        (ncnn_dir / "platform.h").touch()
        (ncnn_root / "src").mkdir()
        (ncnn_root / "src/net.h").touch()

        fake_bin = temporary_root / "bin"
        fake_bin.mkdir()
        cmake_log = temporary_root / "cmake.log"
        example_log = temporary_root / "example.log"
        write_executable(
            fake_bin / "cmake",
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CMAKE_LOG"\n',
        )
        tools = repo / "tools"
        tools.mkdir()
        (tools / "run_example.py").touch()
        write_executable(
            fake_bin / "python",
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$EXAMPLE_LOG"\n',
        )

        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
        env["CMAKE_LOG"] = str(cmake_log)
        env["EXAMPLE_LOG"] = str(example_log)

        detected = subprocess.run(
            [str(quickstart)],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        cmake_args = cmake_log.read_text(encoding="utf-8") if cmake_log.exists() else ""
        example_args = example_log.read_text(encoding="utf-8") if example_log.exists() else ""
        if (
            detected.returncode != 0
            or f"Using model: {model}" not in detected.stdout
            or f"Using ncnn build tree: {ncnn_dir}" not in detected.stdout
            or "-DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF" not in cmake_args
            or f"-DNCNN_INCLUDE_DIR={ncnn_root / 'src'}" not in cmake_args
            or f"-DNCNN_BUILD_INCLUDE_DIR={ncnn_dir}" not in cmake_args
            or f"-DNCNN_LIBRARY={ncnn_dir / 'libncnn.a'}" not in cmake_args
            or "-Dncnn_DIR=" in cmake_args
            or f"--model {model}" not in example_args
            or "--case hunyuan_zimu2" not in example_args
            or "--max-tokens" in example_args
        ):
            print(
                "quickstart did not auto-detect the conventional layout: "
                f"rc={detected.returncode} stdout={detected.stdout!r} "
                f"stderr={detected.stderr!r} cmake={cmake_args!r} example={example_args!r}",
                file=sys.stderr,
            )
            return 1

        package_dir = temporary_root / "installed/lib/cmake/ncnn"
        package_dir.mkdir(parents=True)
        (package_dir / "ncnnConfig.cmake").touch()
        (package_dir / "ncnn.cmake").touch()
        cmake_log.unlink()
        packaged = subprocess.run(
            [
                str(quickstart),
                "--model",
                str(model),
                "--ncnn-dir",
                str(package_dir),
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        package_cmake_args = cmake_log.read_text(encoding="utf-8")
        if (
            packaged.returncode != 0
            or f"Using ncnn package: {package_dir}" not in packaged.stdout
            or f"-Dncnn_DIR={package_dir}" not in package_cmake_args
            or "-DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF" in package_cmake_args
        ):
            print(
                "quickstart did not preserve installed ncnn package discovery: "
                f"rc={packaged.returncode} stdout={packaged.stdout!r} "
                f"stderr={packaged.stderr!r} cmake={package_cmake_args!r}",
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
