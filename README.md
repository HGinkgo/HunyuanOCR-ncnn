<div align="center">
  <h1>HunyuanOCR-ncnn</h1>
  <p>
    基于 <a href="https://huggingface.co/tencent/HunyuanOCR"><strong>HunyuanOCR 1.5</strong></a>
    和 <a href="https://github.com/Tencent/ncnn"><strong>ncnn</strong></a> 的纯 C++17 OCR 推理运行时
  </p>
  <p>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml/badge.svg" alt="Linux CI"></a>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml/badge.svg" alt="Windows CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
    <img src="https://img.shields.io/badge/C%2B%2B-17-f34b7d.svg" alt="C++17">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg" alt="Linux and Windows">
    <img src="https://img.shields.io/badge/backend-CPU%20fp32%20%7C%20Vulkan%20vision%2Ftext%20fp32-4c1.svg" alt="CPU fp32 和 Vulkan vision/text fp32">
  </p>
  <p>
    <a href="https://github.com/Tencent/ncnn/discussions/6808">技术说明</a>
    &nbsp;|&nbsp; <a href="README_en.md">English</a>
  </p>
</div>

---

本项目使用 pnnx 将 Hugging Face 版 HunyuanOCR 导出为 ncnn 子模块，并在 C++ 中完成
图片预处理、动态 vision、prompt、KV cache 解码和 tokenizer 后处理。

> `main` 是 HunyuanOCR 1.5 的 `0.4.0`；`feat/hunyuanocr-1.0` 和 `v0.2.0`
> 保留 HunyuanOCR 1.0 兼容版本。

## 功能特性

- PNG/JPEG 输入和已导出范围内的动态图片尺寸。
- `spotting`、`document` 和自定义 UTF-8 prompt。
- 可复用 C++ runtime、逐 token 流式回调和 JSONL 批量推理。
- 默认 CPU fp32；可选 DFlash、mmap 权重加载和 Vision / Text Vulkan。
- Text Vulkan 通过项目维护的 ncnn 补丁，将 `Gemm_vulkan` 的运行时 `M=1` 路径分派到 GEMV。
- Linux、Windows、UTF-8 路径及命令行支持。
- 公开图片的 token/text 严格测试与 Sanitizer 门禁。

## 快速开始

预转换模型包托管在
[ModelScope](https://modelscope.cn/models/HGinkgo/HunyuanOCR-1.5-ncnn)：

```bash
python -m pip install modelscope
modelscope download \
  --model HGinkgo/HunyuanOCR-1.5-ncnn \
  --local_dir ./hunyuan_ocr_ncnn_model
```

> 该模型包由个人转换和维护，是非官方社区产物，腾讯未对其或本项目提供认可或背书。

脚本会自动发现 `./hunyuan_ocr_ncnn_model` 和相邻的 `../ncnn/build/src`，然后构建并运行示例：

```bash
scripts/quickstart_existing_model.sh
```

模型或 ncnn 位于其他位置时，可分别使用 `--model PATH` 和 `--ncnn-dir PATH` 覆盖。
quickstart 默认只生成 16 个 token 做快速 smoke；完整 OCR 请使用下面的单图命令。

源码仓库不内置模型权重。

## 构建

依赖 CMake 3.18、C++17 编译器和 ncnn。验证 revision 为
`dda2e28bae2a084760361197d87f06e685604e52`。

```bash
cmake -S . -B build -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build build -j
```

使用 Vulkan 前，需对固定 revision 的 ncnn 应用项目补丁，再以 `NCNN_VULKAN=ON`
构建 ncnn：

```bash
python scripts/apply_ncnn_patches.py --ncnn-dir /path/to/ncnn
```

<details>
<summary>直接使用本地 ncnn build 目录</summary>

```bash
cmake -S . -B build \
  -DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF \
  -DNCNN_INCLUDE_DIR=/path/to/ncnn/src \
  -DNCNN_BUILD_INCLUDE_DIR=/path/to/ncnn/build/src \
  -DNCNN_LIBRARY=/path/to/ncnn/build/src/libncnn.a
cmake --build build -j
```

</details>

## 运行与集成

### 单图推理

```bash
./build/hunyuan_ocr_cli \
  --image ./examples/images/hf_demo_tools-dark.png
```

CLI 会自动发现 `./hunyuan_ocr_ncnn_model` 和常见 `assets/` 目录。单图默认使用 `document` prompt，识别文本会在生成过程中直接输出；默认最多生成 8192 个 token，并在 EOS 或尾部重复时提前结束。需要坐标时可显式使用 `--prompt-mode spotting`，也可以使用 `--prompt "只输出图片中的可见文字"` 传入自定义 prompt。

### JSONL 批量推理

每行是一条请求，`prompt_mode` 与 `prompt` 二选一：

```json
{"id":"page-1","image":"images/page-1.png","prompt_mode":"document","max_tokens":256}
```

```bash
./build/hunyuan_ocr_cli \
  --batch-input requests.jsonl \
  --batch-output results.jsonl
```

模型只加载一次，输出顺序与输入一致；失败记录写入 `ok: false`，已有输出需用 `--force`
才会覆盖。

### C++ Runtime

最小可运行集成示例见 [`examples/ocr_main.cpp`](examples/ocr_main.cpp)：

```bash
cmake --build build --target hunyuan_ocr_example
./build/hunyuan_ocr_example ./hunyuan_ocr_ncnn_model ./document.png
```

```cmake
add_subdirectory(path/to/HunyuanOCR-ncnn)
target_link_libraries(my_ocr_app PRIVATE hunyuan_ocr)
```

```cpp
#include "hunyuan_ocr/hunyuan_ocr.h"

hunyuan_ocr::RuntimeError error;
hunyuan_ocr::HunyuanOCR runtime;
if (!runtime.load("./hunyuan_ocr_ncnn_model", {}, &error)) return 1;

hunyuan_ocr::InferenceRequest request;
request.prompt_mode = hunyuan_ocr::PromptMode::Document;
request.max_tokens = 8192;

hunyuan_ocr::InferenceResult result;
if (!runtime.infer_file("document.png", request, &result, &error)) return 2;
```

同一个 runtime 可以顺序处理多次请求；`infer_rgb` 可接收调用方已有的连续 RGB 数据。

## 可选能力

| 能力 | 启用方式 | 说明 |
| --- | --- | --- |
| DFlash | `--dflash` | AR 仍是默认路径；低 acceptance 输入可能更慢，不承诺普遍加速。 |
| mmap 权重 | `--mmap-weights` | 减少加载复制和匿名内存，不缩小模型文件。 |
| Vision Vulkan | `--vision-vulkan` | Vision Encoder 使用 Vulkan；未启用 Text Vulkan 时，文本生成仍为 CPU fp32。 |
| Text Vulkan | `--text-vulkan` | Decoder 和 LM Head 使用 Vulkan；暂不兼容 DFlash。 |

两条 Vulkan 路径均需应用 [`patches/ncnn`](patches/ncnn)，可以单独启用或组合使用：

```bash
./build/hunyuan_ocr_cli --image ./document.png --vision-vulkan --text-vulkan
```

设备可分别通过 `--vision-vulkan-device N` 和 `--text-vulkan-device N` 选择。
C++ 调用方可在 `RuntimeOptions` 中设置对应选项。

## 更多命令

```bash
# 查看 CLI 参数
./build/hunyuan_ocr_cli --help
# 查看项目和 ncnn 版本
./build/hunyuan_ocr_cli --version
# 使用默认示例图执行快速 smoke test
scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model
# 列出内置示例
python tools/run_example.py --list
# 运行单个内置示例
python tools/run_example.py --model ./hunyuan_ocr_ncnn_model --case hf_demo
# 依次运行全部内置示例
python tools/run_examples.py --model ./hunyuan_ocr_ncnn_model
# 运行单例性能测试
python tools/benchmark.py --model ./hunyuan_ocr_ncnn_model --cases hf_demo
```

示例图片及来源位于 [`examples`](examples)，benchmark 字段见
[`tools/README.md`](tools/README.md#benchmark)。

## 当前限制

- 当前模型包使用 `max_pixels=524288`，不包含原版高分辨率路径。
- JPEG 解码器的像素舍入差异可能影响决策敏感图片；稳定复现建议使用 PNG。
- 自定义 prompt 尚未覆盖所有 tokenizer 边界输入。
- Vulkan 是显式可选后端，并依赖固定 ncnn revision 和项目补丁集。
- Text Vulkan 暂不与 DFlash 组合；DFlash 仍使用 CPU fp32 文本路径。
- 当前交付为 CPU/Vulkan fp32；fp16、int8 及 decoder/full-model 量化经评估未达到可发布的质量与性能权衡，因此不提供低精度模型包。

## 许可证

原创代码使用 Apache-2.0。`stb_image.h` 和 picojson 的第三方许可见 [`NOTICE`](NOTICE)。
HunyuanOCR 模型文件遵循 Tencent Hunyuan Community License Agreement。
