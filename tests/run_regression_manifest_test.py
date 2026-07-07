#!/usr/bin/env python3
"""Regression runner manifest behavior tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def write_fake_binary(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

Path = __import__("pathlib").Path
Path(os.environ["ARGV_LOG"]).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
print("  image_grid_thw: [1, 1, 1]")
print("  generated_token_count: 1")
print("  match_fixture_input_ids: true")
print("  match_fixture_position_ids: true")
print("  match_expected_tokens: true")
print("  match_expected_text: true")
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hunyuanocr_run_regression_test_") as tmp_text:
        tmp = Path(tmp_text)
        model = tmp / "model"
        images = tmp / "images"
        fixtures = tmp / "fixtures"
        logs = tmp / "logs"
        fixture_case = fixtures / "custom_prompt_case"
        model.mkdir()
        images.mkdir()
        fixture_case.mkdir(parents=True)
        (images / "image.png").write_bytes(b"not a real image; fake binary does not read it")

        manifest = tmp / "manifest.json"
        prompt = "只输出图片中的可见文字"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "name": "custom_prompt_case",
                        "image": "image.png",
                        "prompt": prompt,
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        fake_binary = tmp / "fake_hunyuan_ocr_cli.py"
        argv_log = tmp / "argv.json"
        write_fake_binary(fake_binary)

        env = os.environ.copy()
        env["ARGV_LOG"] = str(argv_log)
        completed = subprocess.run(
            [
                sys.executable,
                str(repo_root / "tools/run_regression.py"),
                "--binary",
                str(fake_binary),
                "--model",
                str(model),
                "--image-root",
                str(images),
                "--fixture-root",
                str(fixtures),
                "--manifest",
                str(manifest),
                "--log-dir",
                str(logs),
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            env=env,
        )
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode

        argv = json.loads(argv_log.read_text(encoding="utf-8"))
        if "--prompt" not in argv:
            print(f"missing --prompt in argv: {argv}", file=sys.stderr)
            return 1
        if prompt not in argv:
            print(f"missing prompt text in argv: {argv}", file=sys.stderr)
            return 1
        if "--prompt-mode" in argv:
            print(f"unexpected --prompt-mode in argv: {argv}", file=sys.stderr)
            return 1
        if "summary: 1/1 passed" not in completed.stdout:
            print(completed.stdout, end="")
            print("missing expected summary", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
