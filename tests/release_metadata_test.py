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
    tests_cmake_path = root / "tests/CMakeLists.txt"
    require(tests_cmake_path.is_file(), "tests/CMakeLists.txt must own test build definitions")
    tests_cmake = tests_cmake_path.read_text(encoding="utf-8")
    readme_en_path = root / "README_en.md"
    require(readme_en_path.is_file(), "README_en.md must retain the English documentation")
    require(not (root / "README_zh.md").exists(), "README_zh.md must be replaced by the default Chinese README.md")
    readme_en = readme_en_path.read_text(encoding="utf-8")
    readme_zh = (root / "README.md").read_text(encoding="utf-8")
    model_readme = (root / "models/README.md").read_text(encoding="utf-8")
    tools_readme = (root / "tools/README.md").read_text(encoding="utf-8")
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    image_sources = (root / "examples/IMAGE_SOURCES.md").read_text(encoding="utf-8")
    expected_outputs = (root / "examples/EXPECTED_OUTPUTS.md").read_text(encoding="utf-8")

    require("add_subdirectory(tests)" in cmake, "root CMake must delegate test definitions")
    require("add_test(" not in cmake, "root CMake must not register individual tests")
    require("add_executable(hunyuan_ocr_api_test" in tests_cmake,
            "tests CMake must own C++ test targets")
    require("add_test(NAME cli_options" in tests_cmake,
            "tests CMake must own Python integration tests")

    public_docs = [
        root / "README.md",
        root / "README_en.md",
        root / "NOTICE",
        root / "examples/EXPECTED_OUTPUTS.md",
        root / "examples/IMAGE_SOURCES.md",
        root / "examples/README.md",
        root / "export/README.md",
        root / "models/README.md",
        root / "tools/README.md",
    ]
    public_docs.extend(sorted((root / "docs").rglob("*.md")))
    for path in public_docs:
        require(
            "/root/" not in path.read_text(encoding="utf-8"),
            f"public documentation must not contain a local /root path: {path.relative_to(root)}",
        )

    require(re.search(r"project\(HunyuanOCR_ncnn\s+VERSION 0\.4\.0", cmake) is not None, "CMake version must be 0.4.0")
    for text, label in ((readme_en, "README_en"), (readme_zh, "README")):
        require(REVISION in text, f"{label} must record the fixed checkpoint revision")
        require("Transformers 5.13.0" in text, f"{label} must record Transformers 5.13.0")
        require("v0.2.0" in text, f"{label} must retain the frozen HunyuanOCR 1.0 tag")
        require("--batch-input" in text, f"{label} must document JSONL batch input")
        require("--batch-output" in text, f"{label} must document JSONL batch output")
        require("infer_file" in text, f"{label} must document the reusable C++ API")
        require("add_subdirectory" in text, f"{label} must document source-tree library use")
        competition_rows = [line for line in text.splitlines() if line.startswith("|") and "ncnn_llm" in line]
        require(len(competition_rows) == 1, f"{label} must contain one ncnn_llm competition row")
        require("picojson" in competition_rows[0], f"{label} competition dependency row must disclose picojson")
    require("picojson" in notice and "BSD 2-Clause" in notice,
            "NOTICE must attribute the vendored JSON parser")
    require("HunyuanOCR 1.5 preview" in readme_en, "README_en must mark HunyuanOCR 1.5 as preview")
    for text, label in ((readme_en, "README_en"), (readme_zh, "README")):
        require("main" in text, f"{label} must identify the HunyuanOCR 1.5 default branch")
        require("feat/hunyuanocr-1.0" in text, f"{label} must identify the preserved HunyuanOCR 1.0 branch")
    require("\n## Status\n" not in readme_en, "README_en must not duplicate highlights in a Status section")
    require("\n## 当前状态\n" not in readme_zh, "README must not duplicate current delivery details")
    require('<a href="README_en.md">English</a>' in readme_zh,
            "README must link to the English documentation")
    require('<a href="README.md">中文说明</a>' in readme_en,
            "README_en must link to the default Chinese documentation")
    require("Windows build and packaged-model validation passed." in readme_en, "README_en must retain Windows validation evidence")
    require("Windows 构建和带模型验证均已通过。" in readme_zh, "README must retain Windows validation evidence")
    require("MSVC" not in readme_en and "UCRT64" not in readme_en, "README_en Windows status must not expose toolchain details")
    require("MSVC" not in readme_zh and "UCRT64" not in readme_zh, "README Windows status must not expose toolchain details")
    require("validation pending" not in readme_en and "still pending" not in readme_en, "README_en contains stale pending status")
    require("尚待验证" not in readme_zh and "尚待复核" not in readme_zh, "README contains stale pending status")
    require("`0.4.0` preview" in model_readme, "model README must identify the untagged 0.4.0 preview")
    require("Experimental DFlash" in readme_en, "README_en must label DFlash as experimental")
    require("实验性 DFlash" in readme_zh, "README must label DFlash as experimental")
    require("https://github.com/nihui/ncnn_llm" in readme_en, "README_en must reference ncnn_llm")
    require("https://github.com/nihui/ncnn_llm" in readme_zh, "README must reference ncnn_llm")
    require("\n## Competition Coverage\n" in readme_en, "README_en must map the competition requirements")
    require("\n## 比赛要求覆盖\n" in readme_zh, "README must map the competition requirements")
    for text, label, quick_start, competition, advanced in (
        (
            readme_en,
            "README_en",
            "## Quick Start",
            "## Competition Coverage",
            "## Advanced Engineering",
        ),
        (readme_zh, "README", "## 快速开始", "## 比赛要求覆盖", "## 扩展能力"),
    ):
        require(
            text.count(quick_start) == 1,
            f"{label} must contain exactly one quick-start section",
        )
        require(
            text.index(quick_start) < text.index(competition) < text.index(advanced),
            f"{label} must lead with user onboarding before competition and engineering details",
        )
    require("CI covers Linux and Windows builds and lightweight tests" in readme_en,
            "README_en must distinguish CI from packaged-model validation")
    require("packaged-model validation was completed separately on both platforms" in readme_en,
            "README_en must retain separate packaged-model validation evidence")
    require("CI 覆盖 Linux、Windows 构建和轻量测试" in readme_zh,
            "README must distinguish CI from packaged-model validation")
    require("带模型验证已在两个平台独立完成" in readme_zh,
            "README must retain separate packaged-model validation evidence")
    require("\n## Advanced Engineering\n" in readme_en, "README_en must identify advanced engineering work")
    require("\n## 扩展能力\n" in readme_zh, "README must identify extended capabilities")
    require("AR remains the default" in readme_en, "README_en must retain AR as the default path")
    require("AR 仍是默认路径" in readme_zh, "README must retain AR as the default path")
    require("may be slower" in readme_en, "README_en must disclose low-acceptance slowdown")
    require("可能更慢" in readme_zh, "README must disclose low-acceptance slowdown")
    require("DFlash" in model_readme, "model README must document optional DFlash artifacts")
    require("DFlash" in tools_readme, "tools README must document DFlash packaging")
    for text, label in ((readme_en, "README_en"), (readme_zh, "README")):
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
