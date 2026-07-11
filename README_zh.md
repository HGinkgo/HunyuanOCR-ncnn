# HunyuanOCR-ncnn

[![Linux CI](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml)
[![Windows Compile](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml/badge.svg)](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml)

Tencent HunyuanOCR 的 C++17/ncnn 推理运行时。

技术报告：[Tencent ncnn Discussion #6808](https://github.com/Tencent/ncnn/discussions/6808)

本仓库使用 pnnx 将 Hugging Face 版 HunyuanOCR 导出为 ncnn 子模块，并在 C++ 中跑通完整 OCR 推理链路。

> **HunyuanOCR 1.5 预览版本（`0.3.0`）：** 当前开发分支固定 checkpoint
> revision `9e01f897bf8956f77a80c350dc0491d6bbbd43e6`，严格参考使用
> Transformers 5.13.0、CPU fp32 和 eager attention。Linux 与 Windows 验证均
> 已通过，28 个公开 case 的前 128 个生成 token 及对应 decode 文本严格一致；
> `v0.2.0` 继续作为冻结的 HunyuanOCR 1.0 版本。

| 开发线 | 模型 | 状态 |
| --- | --- | --- |
| `main` | HunyuanOCR 1.5 | `0.3.0` development/preview |
| `feat/hunyuanocr-1.0` | HunyuanOCR 1.0 | 保留的兼容分支 |
| `v0.2.0` | HunyuanOCR 1.0 | 冻结版本 |

## 当前交付能力

- C++17 端到端完成 PNG/JPEG 图片输入到 OCR 文本输出。
- 支持已导出范围内的不同图片尺寸，并保留 fixed-grid 回退包。
- 带 KV cache 的文本解码、贪心解码和重复惩罚。
- 内置 `spotting` / `document` 两种模式，也支持自定义 `--prompt` 文本。
- Windows CLI 全链路支持 UTF-8 prompt、模型路径、图片路径和 fixture 路径。
- CMake 构建，Linux / Windows 均已验证；运行时不依赖 Python。
- 28 个公开 case 在已验证的 128-token 窗口内与 HunyuanOCR 1.5 PyTorch fp32
  reference 的 token/text 一致。

## 常用入口

| 需求 | 入口 |
| --- | --- |
| 构建并跑一张图 | `scripts/quickstart_existing_model.sh` |
| 打包转换产物 | `tools/package_model.py` |
| 从 HF 权重导出 | `export/README.md` |
| 运行示例图 | `tools/run_example.py`, `tools/run_examples.py` |
| 严格回归 | `tools/run_regression.py` |
| 性能测试 | `tools/benchmark.py`, `benchmark/README.md` |
| 模型目录协议 | `models/README.md`, `models/model.json.example` |

## 当前状态

| 项目 | 状态 |
| --- | --- |
| Linux | 本地完成 HunyuanOCR 1.5 构建、CTest 和 28-case 128-token 回归 |
| Windows | 构建和带模型验证通过 |
| 输入 | PNG/JPEG 图片 |
| 输出 | OCR 文本 |
| 验证 | 28 个 case 的前 128 个生成 token/text 与 1.5 PyTorch fp32 reference 一致 |
| 精度 | fp32 ncnn 路径 |
| Prompt | 内置 `spotting` / `document` 模式，也支持自定义 `--prompt` 文本 |
| Vision | 支持已导出范围内的不同图片尺寸，并保留 fixed-grid 回退包 |

当前已验证配置使用 `max_pixels=524288`。就运行方式而言，dynamic vision 包可覆盖已导出范围内的不同图片尺寸；fixed-grid 包主要作为兼容回退。模型目录和字段说明见 `models/README.md`。

当前版本暂不覆盖原版高分辨率路径。

对于对 JPEG 解码舍入敏感的 case，严格回归使用规范化无损 PNG。运行时仍
支持 JPEG，但 Pillow/libjpeg-turbo 与 `stb_image` 对有损 JPEG 的解码像素可能
存在微小差异，因此不承诺任意 JPEG 都能跨解码器保持 token 完全一致。

## 构建

如果已经准备好打包后的 `hunyuan_ocr_ncnn_model/`，可以用最短路径构建并跑一张示例图：

```bash
scripts/quickstart_existing_model.sh \
  --model ./hunyuan_ocr_ncnn_model \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn
```

如果已经构建完成，只想跑一张图做 smoke test：

```bash
scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model
```

依赖：

- CMake 3.18 或更新版本
- 支持 C++17 的编译器
- ncnn `20260106` 或更新版本；验证固定 revision 为
  `244f30c8b995d5b2cf57b59950596490c68813d6`

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

Windows 构建和带模型验证均已通过。CLI 从宽字符命令行读取参数，支持中文等
Unicode prompt 和模型、图片、fixture 路径；控制台输入输出统一使用 UTF-8。

## 准备模型目录

运行 OCR 需要一个打包后的 ncnn 模型目录：

```text
hunyuan_ocr_ncnn_model/
  model.json
  tokenizer/
  text_embed/
  text_decoder/
  lm_head/
  vision/
```

如果已经有导出的 ncnn 产物，可以用工具打包：

```bash
python tools/package_model.py \
  --workspace <workspace> \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

这里 `<workspace>` 指包含 `models/tokenizer/` 和 `models/export/` 的工作目录。
如果需要 v0.1 fixed-grid 包，使用 `--vision-backend fixed`；如果需要同时包含 dynamic vision 和 fixed-grid fallback，使用 `--vision-backend both`。
`models/README.md` 说明了 `model.json` 字段和 dynamic/fixed vision backend 的选择规则。

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

## 从 HF 模型导出

导出脚本位于 `export/`，输出目录结构与 `tools/package_model.py` 兼容：

```bash
python export/export_all.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --workspace .

python tools/package_model.py \
  --workspace . \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

Linux 下也可以使用封装脚本：

```bash
scripts/export_and_package_linux.sh \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn \
  --output ./hunyuan_ocr_ncnn_model \
  --copy
```

模块级命令见 `export/README.md`。

## 完整回归

准备好 baseline/export 产生的 fixture 后，可以运行完整 token/text 回归：

```bash
python tools/run_regression.py \
  --package \
  --package-vision-backend dynamic
```

期望摘要：

```text
summary: 28/28 passed
```

该回归会比较 prompt ids、position ids、generated token ids 和最终 decode 文本。

## 性能测试

构建 CLI 并准备好打包模型后：

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases hf_demo \
  --repeat 3 \
  --warmup 1 \
  --max-tokens 64
```

输出包含图片预处理、vision、prompt 组装、text embedding、prefill、增量 decode、总耗时和 decode token/s。更多说明见 `benchmark/README.md`。

## 当前限制

- dynamic vision backend 已在 28 个 case 的前 128 个生成 token 上完成 token/text 回归验证，验证口径为 `max_pixels=524288`。
- HunyuanOCR 1.5 仍是绑定固定 checkpoint revision 和已验证 128-token 窗口的 preview。
- 有损 JPEG 的跨解码器舍入差异可能改变决策敏感样本的生成 token；严格对齐时使用规范化无损输入。
- 自定义 prompt 已接入 C++ tokenizer encode；更多 tokenizer 边界还需要继续补充 HF 对齐测试。
- 当前交付范围使用 `max_pixels=524288`，不包含原版高分辨率路径。
- 公开示例脚本只验证端到端运行；严格 token/text 对齐由 fixture 回归验证。

## 许可证

本仓库原创代码使用 Apache-2.0。`third_party/stb_image.h` 保留上游 MIT/Public Domain 许可说明。

HunyuanOCR 模型文件遵循 Tencent Hunyuan Community License Agreement。
