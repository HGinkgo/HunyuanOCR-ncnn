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
    <img src="https://img.shields.io/badge/backend-CPU%20fp32%20%7C%20Vulkan%20vision%20fp32-4c1.svg" alt="CPU fp32 和 Vulkan vision fp32">
  </p>
  <p>
    Tencent/ncnn 开源活动作品 <a href="https://github.com/Tencent/ncnn/discussions/6808">#6808</a>
    &nbsp;|&nbsp; <a href="README_en.md">English</a>
  </p>
</div>

---

本仓库使用 pnnx 将 Hugging Face 版 HunyuanOCR 导出为 ncnn 子模块，并在 C++ 中跑通完整 OCR 推理链路。

> **HunyuanOCR 1.5 预览版本（`0.4.0`）：** `main` 与下方 ModelScope 模型包配套使用；
> `v0.2.0` 继续作为冻结的 HunyuanOCR 1.0 版本。

| 开发线 | 模型 | 状态 |
| --- | --- | --- |
| `main` | HunyuanOCR 1.5 | `0.4.0` development/preview |
| `feat/hunyuanocr-1.0` | HunyuanOCR 1.0 | 保留的兼容分支 |
| `v0.2.0` | HunyuanOCR 1.0 | 冻结版本 |

## 快速开始

预转换的 ncnn 运行时模型包独立托管在
[ModelScope](https://modelscope.cn/models/HGinkgo/HunyuanOCR-1.5-ncnn)：

```bash
python -m pip install modelscope
modelscope download \
  --model HGinkgo/HunyuanOCR-1.5-ncnn \
  --local_dir ./hunyuan_ocr_ncnn_model
```

> 该模型包由个人转换和维护，属于非官方社区产物，并非腾讯官方发布；腾讯未对该模型包
> 或本项目提供认可或背书。

下载完成后，可以用最短路径构建并跑一张示例图：

```bash
scripts/quickstart_existing_model.sh \
  --model ./hunyuan_ocr_ncnn_model \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn
```

如果已经构建完成，只想跑一张图做 smoke test：

```bash
scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model
```

源码仓库不内置模型权重，直接下载上述运行时模型包即可。

## 常用入口

| 需求 | 入口 |
| --- | --- |
| 下载预转换模型 | [ModelScope](https://modelscope.cn/models/HGinkgo/HunyuanOCR-1.5-ncnn) |
| 构建并跑一张图 | `scripts/quickstart_existing_model.sh` |
| 运行示例图 | `tools/run_example.py`, `tools/run_examples.py` |
| JSONL 批量推理 | `--batch-input`, `--batch-output` |
| C++ 接入 | `hunyuan_ocr` 静态库 |
| 性能测试 | `tools/benchmark.py`, `tools/README.md#benchmark` |

## 功能特性

- 支持 PNG/JPEG 和已导出范围内的动态图片尺寸。
- 内置 `spotting` / `document` 两种模式，也支持自定义 UTF-8 `--prompt`。
- 可复用 C++ runtime 连续处理多张图片，无需重复加载模型。
- JSONL 批量推理保持输入顺序，并为每条失败记录返回独立错误。
- Linux 和 Windows CLI 均支持 UTF-8 prompt、模型路径和图片路径。
- 可选 DFlash speculative decoding、只读 mmap 权重加载和 fp32 Vulkan vision。

## DFlash

`--dflash` 可以为 greedy generation 显式启用 DFlash speculative decoder，
ModelScope 模型包已包含所需文件。AR 仍是默认路径，运行时不会自动切换解码方式。

性能收益取决于输入，低 acceptance 输入可能更慢，因此 DFlash 不作为通用加速承诺。

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode document \
  --dflash
```

## 可选 mmap 权重加载

`--mmap-weights` 让 vision、text 和 DFlash ncnn 网络从只读文件映射加载权重，
默认加载方式保持不变。该选项可以减少模型加载复制和匿名内存占用。

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode document \
  --mmap-weights
```

C++ 调用方可以设置 `RuntimeOptions::mmap_weights = true`。该选项不会缩小模型文件，
也不改变推理数值逻辑。

## 构建

依赖：

- CMake 3.18 或更新版本
- 支持 C++17 的编译器
- ncnn `20260106` 或更新版本；验证固定 revision 为
  `dda2e28bae2a084760361197d87f06e685604e52`

默认 CPU 构建可以直接使用未修改的固定 ncnn checkout。可选的 fp32 Vulkan
vision 后端使用本项目维护的 [`patches/ncnn`](patches/ncnn) 补丁集：

```bash
python scripts/apply_ncnn_patches.py --ncnn-dir /path/to/ncnn
```

如果已经安装 ncnn CMake package：

```bash
cmake -S . -B build -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build build -j
```

如果使用本地 ncnn build 目录：

```bash
cmake -S . -B build \
  -DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF \
  -DNCNN_INCLUDE_DIR=/path/to/ncnn/src \
  -DNCNN_BUILD_INCLUDE_DIR=/path/to/ncnn/build/src \
  -DNCNN_LIBRARY=/path/to/ncnn/build/src/libncnn.a
cmake --build build -j
```

基础检查：

```bash
./build/hunyuan_ocr_cli --help
./build/hunyuan_ocr_cli --version
```

构建补丁版 ncnn SDK 后，可以让 vision encoder 使用 Vulkan：

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode spotting \
  --vision-vulkan \
  --vision-vulkan-device 0
```

当前仅 vision encoder 使用 Vulkan；text generation 仍使用原有 CPU fp32
路径。

Windows CLI 支持中文等 Unicode prompt、模型路径和图片路径，控制台输入输出统一
使用 UTF-8。

## 可复用 C++ Runtime

`hunyuan_ocr` 静态库目标持有已加载的模型，可以连续处理多次请求而不重新加载
ncnn 网络。源码方式接入如下：

```cmake
add_subdirectory(path/to/HunyuanOCR-ncnn)
target_link_libraries(my_ocr_app PRIVATE hunyuan_ocr)
```

```cpp
#include "hunyuan_ocr/hunyuan_ocr.h"

hunyuan_ocr::RuntimeOptions options;
options.num_threads = 8;

hunyuan_ocr::RuntimeError error;
hunyuan_ocr::HunyuanOCR runtime;
if (!runtime.load("./hunyuan_ocr_ncnn_model", options, &error)) return 1;

hunyuan_ocr::InferenceRequest request;
request.prompt_mode = hunyuan_ocr::PromptMode::Document;
request.max_tokens = 128;

hunyuan_ocr::InferenceResult result;
if (!runtime.infer_file("document.png", request, &result, &error)) return 2;
```

调用方已经持有解码后像素时，可以使用接收连续 RGB 字节的 `infer_rgb`。同一个
Runtime 按顺序处理请求，不保证并发调用安全；需要并发时可创建多个实例，但会占用
多份模型内存。每次请求结束后，图片张量、vision features 和 KV cache 都会释放。

## JSONL 批量推理

批量模式只加载一次模型，并为每个物理输入行按顺序写出一条结果：

```json
{"id":"page-1","image":"images/page-1.png","prompt_mode":"document","max_tokens":256}
{"id":"page-2","image":"images/page-2.png","prompt":"只输出发票号码"}
```

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --batch-input requests.jsonl \
  --batch-output results.jsonl \
  --num-threads 8
```

`id` 必须非空且唯一。每条记录必须在 `prompt_mode`（`spotting` 或 `document`）
与 `prompt` 中二选一，`max_tokens` 可以省略。相对图片路径以输入 JSONL 所在目录
为基准。无效或推理失败的记录会写成 `ok: false`，后续记录仍继续执行；只要存在
失败记录，进程最终返回非零退出码。输出文件已存在时默认拒绝覆盖，显式传入
`--force` 才会重写。

## 运行示例

查看内置示例图：

```bash
python tools/run_example.py --list
```

运行单张图片：

```bash
python tools/run_example.py \
  --model ./hunyuan_ocr_ncnn_model \
  --case hf_demo
```

运行单张图片并传入自定义 prompt：

```bash
python tools/run_example.py \
  --model ./hunyuan_ocr_ncnn_model \
  --case hf_demo \
  --prompt "只输出图片中的可见文字"
```

运行全部示例图：

```bash
python tools/run_examples.py \
  --model ./hunyuan_ocr_ncnn_model
```

示例图片位于 `examples/images/`，来源记录在 `examples/IMAGE_SOURCES.md`。

## 性能测试

构建 CLI 并下载模型后：

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases hf_demo \
  --repeat 3 \
  --warmup 1 \
  --max-tokens 64
```

输出包含图片预处理、vision、prompt 组装、text embedding、prefill、增量 decode、总耗时和 decode token/s。更多说明见 `tools/README.md#benchmark`。

## 当前限制

- HunyuanOCR 1.5 当前仍是 `0.4.0` preview，请使用 README 链接的配套模型包。
- 当前模型包使用 `max_pixels=524288`，不包含原版高分辨率路径。
- JPEG 解码器之间的像素舍入差异可能影响少数决策敏感图片；需要稳定复现时建议使用 PNG。
- 自定义 prompt 已支持 UTF-8 文本，但尚未覆盖所有 tokenizer 边界输入。
- Vulkan 只用于 vision encoder，text generation 仍使用 CPU fp32。

## 许可证

本仓库原创代码使用 Apache-2.0。`third_party/stb_image.h` 保留上游 MIT/Public Domain
许可说明，vendored picojson 保留 BSD 2-Clause 许可说明。

HunyuanOCR 模型文件遵循 Tencent Hunyuan Community License Agreement。
