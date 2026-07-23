#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


VERSION = "0.4.0"
REVISION = "9e01f897bf8956f77a80c350dc0491d6bbbd43e6"
MODELSCOPE_MODEL = "https://modelscope.cn/models/HGinkgo/HunyuanOCR-1.5-ncnn"
DISCUSSION = "https://github.com/Tencent/ncnn/discussions/6808"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_commented_commands(text: str, heading: str, label: str) -> None:
    section_start = text.index(heading)
    section_end = text.find("\n## ", section_start + len(heading))
    section = text[section_start:section_end if section_end >= 0 else len(text)]
    block = re.search(r"```bash\n(.*?)\n```", section, flags=re.DOTALL)
    require(block is not None, f"{label} must contain a bash command block")

    lines = [line for line in block.group(1).splitlines() if line.strip()]
    commands = 0
    for index, line in enumerate(lines):
        if line.startswith("# "):
            continue
        commands += 1
        require(
            index > 0 and lines[index - 1].startswith("# "),
            f"{label} command must have an immediately preceding comment: {line}",
        )
    require(commands >= 7, f"{label} must retain the common command entries")


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
    examples_readme = (root / "examples/README.md").read_text(encoding="utf-8")

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
        text = path.read_text(encoding="utf-8")
        require(
            "/root/" not in text,
            f"public documentation must not contain a local /root path: {path.relative_to(root)}",
        )
        require(
            "fixed-grid fallback" not in text.lower(),
            f"public documentation must not advertise removed fixed-grid fallback: {path.relative_to(root)}",
        )

    require(re.search(r"project\(HunyuanOCR_ncnn\s+VERSION 0\.4\.0", cmake) is not None, "CMake version must be 0.4.0")
    for text, label in ((readme_en, "README_en"), (readme_zh, "README")):
        require("v0.2.0" in text, f"{label} must retain the frozen HunyuanOCR 1.0 tag")
        require("--batch-input" in text, f"{label} must document JSONL batch input")
        require("--batch-output" in text, f"{label} must document JSONL batch output")
        require("infer_file" in text, f"{label} must document the reusable C++ API")
        require("add_subdirectory" in text, f"{label} must document source-tree library use")
        require(MODELSCOPE_MODEL in text, f"{label} must link the pre-converted ModelScope model")
        require("modelscope download" in text, f"{label} must document the ModelScope download command")
        require(DISCUSSION in text, f"{label} must link the technical Discussion")
        require("--dflash" in text, f"{label} must document optional DFlash use")
        require("--mmap-weights" in text, f"{label} must document optional mmap loading")
        require("--vision-vulkan" in text, f"{label} must document optional Vulkan vision use")
        require("--text-vulkan" in text, f"{label} must document optional Vulkan text use")
        require("scripts/apply_ncnn_patches.py" in text,
                f"{label} must document the required ncnn Vulkan patch step")
        for developer_entry in ("export/export_all.py", "tools/package_model.py", "tools/run_regression.py"):
            require(developer_entry not in text,
                    f"{label} must keep developer workflow out of the user README: {developer_entry}")
    require("picojson" in notice and "BSD 2-Clause" in notice,
            "NOTICE must attribute the vendored JSON parser")
    for text, label in ((readme_en, "README_en"), (readme_zh, "README"), (model_readme, "model README")):
        require("0.4.0" in text, f"{label} must identify version 0.4.0")
        require("preview" not in text.lower(), f"{label} must not mark version 0.4.0 as preview")
    for text, label in ((readme_en, "README_en"), (readme_zh, "README")):
        require(
            "backend-CPU%20fp32%20%7C%20Vulkan%20vision%2Ftext%20fp32" in text,
            f"{label} must retain the CPU/Vulkan vision-text backend badge",
        )
        require("main" in text, f"{label} must identify the HunyuanOCR 1.5 default branch")
        require("feat/hunyuanocr-1.0" in text, f"{label} must identify the preserved HunyuanOCR 1.0 branch")
    require("\n## Status\n" not in readme_en, "README_en must not duplicate highlights in a Status section")
    require("\n## 当前状态\n" not in readme_zh, "README must not duplicate current delivery details")
    require('<a href="README_en.md">English</a>' in readme_zh,
            "README must link to the English documentation")
    require('<a href="README.md">中文说明</a>' in readme_en,
            "README_en must link to the default Chinese documentation")
    require("MSVC" not in readme_en and "UCRT64" not in readme_en, "README_en Windows status must not expose toolchain details")
    require("MSVC" not in readme_zh and "UCRT64" not in readme_zh, "README Windows status must not expose toolchain details")
    require("validation pending" not in readme_en and "still pending" not in readme_en, "README_en contains stale pending status")
    require("尚待验证" not in readme_zh and "尚待复核" not in readme_zh, "README contains stale pending status")
    require("Experimental DFlash" not in readme_en, "README_en must not label DFlash as experimental")
    require("实验性 DFlash" not in readme_zh, "README must not label DFlash as experimental")
    require("\n## Competition Coverage\n" not in readme_en,
            "README_en must leave competition details to the Discussion")
    require("\n## 比赛要求覆盖\n" not in readme_zh,
            "README must leave competition details to the Discussion")
    for text, label, headings, subsections in (
        (
            readme_en,
            "README_en",
            [
                "## Features",
                "## Quick Start",
                "## Build",
                "## Run and Integrate",
                "## Optional Capabilities",
                "## More Commands",
                "## Limitations",
                "## License",
            ],
            [
                "### Single-image inference",
                "### JSONL batch inference",
                "### C++ runtime",
            ],
        ),
        (
            readme_zh,
            "README",
            [
                "## 功能特性",
                "## 快速开始",
                "## 构建",
                "## 运行与集成",
                "## 可选能力",
                "## 更多命令",
                "## 当前限制",
                "## 许可证",
            ],
            [
                "### 单图推理",
                "### JSONL 批量推理",
                "### C++ Runtime",
            ],
        ),
    ):
        require(
            re.findall(r"^## .+$", text, flags=re.MULTILINE) == headings,
            f"{label} must use the compact user-facing section structure",
        )
        require(
            all(section in text for section in subsections),
            f"{label} must retain single-image, JSONL, and C++ integration paths",
        )
        require(
            len(text.splitlines()) <= 220,
            f"{label} must stay within the 220-line compact README budget",
        )
    require("\n## Advanced Engineering\n" not in readme_en,
            "README_en must present user features instead of engineering audit details")
    require("\n## 扩展能力\n" not in readme_zh,
            "README must present user features instead of engineering audit details")
    require("AR remains the default" in readme_en, "README_en must retain AR as the default path")
    require("AR 仍是默认路径" in readme_zh, "README must retain AR as the default path")
    require("may be slower" in readme_en, "README_en must disclose low-acceptance slowdown")
    require("可能更慢" in readme_zh, "README must disclose low-acceptance slowdown")
    require_commented_commands(readme_en, "## More Commands", "README_en More Commands")
    require_commented_commands(readme_zh, "## 更多命令", "README 更多命令")
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
    counts = [re.search(r"\| ([0-9]+) \| ([0-9]+) \|$", row) for row in rows]
    require(all(match is not None for match in counts), "expected output rows must record token and character counts")
    token_counts = [int(match.group(1)) for match in counts if match is not None]
    require(all(0 < count <= 512 for count in token_counts), "expected output token counts must fit the 512-token window")
    require(any(count == 512 for count in token_counts), "expected outputs must cover the 512-token safety limit")
    require("maximum coordinate delta of 3" in expected_outputs,
            "expected outputs must disclose the bbox-only numerical boundary")
    require("--package-vision-backend" not in examples_readme,
            "example regression command must use supported run_regression options")
    require("3 coordinate units" in examples_readme and "3 pixels" not in examples_readme,
            "bbox boundary must be reported in coordinate units, not image pixels")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
