#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


VERSION = "0.4.0"
REVISION = "9e01f897bf8956f77a80c350dc0491d6bbbd43e6"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_zh = (root / "README_zh.md").read_text(encoding="utf-8")
    model_readme = (root / "models/README.md").read_text(encoding="utf-8")
    tools_readme = (root / "tools/README.md").read_text(encoding="utf-8")
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    image_sources = (root / "examples/IMAGE_SOURCES.md").read_text(encoding="utf-8")
    expected_outputs = (root / "examples/EXPECTED_OUTPUTS.md").read_text(encoding="utf-8")

    require(re.search(r"project\(HunyuanOCR_ncnn\s+VERSION 0\.4\.0", cmake) is not None, "CMake version must be 0.4.0")
    for text, label in ((readme, "README"), (readme_zh, "README_zh")):
        require(REVISION in text, f"{label} must record the fixed checkpoint revision")
        require("Transformers 5.13.0" in text, f"{label} must record Transformers 5.13.0")
        require("v0.2.0" in text, f"{label} must retain the frozen HunyuanOCR 1.0 tag")
        require("--batch-input" in text, f"{label} must document JSONL batch input")
        require("--batch-output" in text, f"{label} must document JSONL batch output")
        require("infer_file" in text, f"{label} must document the reusable C++ API")
        require("add_subdirectory" in text, f"{label} must document source-tree library use")
    require("picojson" in notice and "BSD 2-Clause" in notice,
            "NOTICE must attribute the vendored JSON parser")
    require("HunyuanOCR 1.5 preview" in readme, "README must mark HunyuanOCR 1.5 as preview")
    for text, label in ((readme, "README"), (readme_zh, "README_zh")):
        require("main" in text, f"{label} must identify the HunyuanOCR 1.5 default branch")
        require("feat/hunyuanocr-1.0" in text, f"{label} must identify the preserved HunyuanOCR 1.0 branch")
    require("\n## Status\n" not in readme, "README must not duplicate highlights in a Status section")
    require("\n## 当前状态\n" not in readme_zh, "README_zh must not duplicate current delivery details")
    require("Windows build and packaged-model validation passed." in readme, "README must retain Windows validation evidence")
    require("Windows 构建和带模型验证均已通过。" in readme_zh, "README_zh must retain Windows validation evidence")
    require("MSVC" not in readme and "UCRT64" not in readme, "README Windows status must not expose toolchain details")
    require("MSVC" not in readme_zh and "UCRT64" not in readme_zh, "README_zh Windows status must not expose toolchain details")
    require("validation pending" not in readme and "still pending" not in readme, "README contains stale pending status")
    require("尚待验证" not in readme_zh and "尚待复核" not in readme_zh, "README_zh contains stale pending status")
    require("`0.4.0` preview" in model_readme, "model README must identify the untagged 0.4.0 preview")
    require("Experimental DFlash" in readme, "README must label DFlash as experimental")
    require("实验性 DFlash" in readme_zh, "README_zh must label DFlash as experimental")
    require("https://github.com/nihui/ncnn_llm" in readme, "README must reference ncnn_llm")
    require("https://github.com/nihui/ncnn_llm" in readme_zh, "README_zh must reference ncnn_llm")
    require("\n## Competition Coverage\n" in readme, "README must map the competition requirements")
    require("\n## 比赛要求覆盖\n" in readme_zh, "README_zh must map the competition requirements")
    require("\n## Advanced Engineering\n" in readme, "README must identify advanced engineering work")
    require("\n## 高阶工程能力\n" in readme_zh, "README_zh must identify advanced engineering work")
    require("AR remains the default" in readme, "README must retain AR as the default path")
    require("AR 仍是默认路径" in readme_zh, "README_zh must retain AR as the default path")
    require("may be slower" in readme, "README must disclose low-acceptance slowdown")
    require("可能更慢" in readme_zh, "README_zh must disclose low-acceptance slowdown")
    require("DFlash" in model_readme, "model README must document optional DFlash artifacts")
    require("DFlash" in tools_readme, "tools README must document DFlash packaging")
    for text, label in ((readme, "README"), (readme_zh, "README_zh")):
        for image in re.findall(r"--image\s+\./examples/images/(\S+)", text):
            require((root / "examples" / "images" / image).is_file(), f"{label} image does not exist: {image}")
    for image in ("hunyuan_vis_art_16.png", "hunyuan_ie_parallel.png"):
        require(image in image_sources, f"image source missing canonical PNG: {image}")

    require(REVISION in expected_outputs, "expected outputs must record the checkpoint revision")
    require("Transformers 5.13.0" in expected_outputs, "expected outputs must record Transformers 5.13.0")
    rows = [line for line in expected_outputs.splitlines() if line.startswith("| `")]
    require(len(rows) == 28, f"expected outputs must contain 28 cases, got {len(rows)}")
    require(all(re.search(r"\| 128 \| [0-9]+ \|$", row) for row in rows), "all expected output rows must use 128 tokens")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
