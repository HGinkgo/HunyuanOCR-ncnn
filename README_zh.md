# HunyuanOCR-ncnn

[![Linux CI](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml)
[![Windows Compile](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml/badge.svg)](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml)

Tencent HunyuanOCR 的 C++/ncnn 推理运行时。

技术报告：[Tencent ncnn Discussion #6808](https://github.com/Tencent/ncnn/discussions/6808)

本项目使用 pnnx 将 HunyuanOCR 拆分为 ncnn 子模块，并在 C++17 中串起图片解码、图像预处理、dynamic/fixed-grid vision 推理、KV cache 文本解码、lm head 和 tokenizer decode。

## 当前交付能力

- C++17 端到端完成 PNG/JPEG 图片输入到 OCR 文本输出。
- dynamic vision backend，并保留 fixed-grid fallback。
- KV cache text decoder、greedy decode 和 repetition penalty。
- 内置 `spotting` / `document` prompt，也支持自定义 `--prompt` 文本。
- CMake 构建，Linux / Windows 均已验证；运行时不依赖 Python。
- 28 张示例图与 PyTorch fp32 reference 的 token/text 一致。

## 常用入口

| 需求 | 入口 |
| --- | --- |
| 构建并跑一张图 | `scripts/quickstart_existing_model.sh` |
| 打包转换产物 | `tools/package_model.py` |
| 从 HF 权重导出 | `export/README.md` |
| 运行示例图 | `tools/run_example.py`, `tools/run_examples.py` |
| strict regression | `tools/run_regression.py` |
| 性能测试 | `tools/benchmark.py`, `benchmark/README.md` |
| 模型目录协议 | `models/README.md`, `models/model.json.example` |

## 当前状态

| 项目 | 状态 |
| --- | --- |
| Linux | 本地完成 CMake 构建和 28 图运行回归 |
| Windows CI | GitHub Actions 只做 MSVC 编译验证 |
| Windows 实机 | 本地 Windows 机器完成带模型运行验证 |
| 输入 | PNG/JPEG 图片 |
| 输出 | OCR 文本 |
| 验证 | 28 张示例图与 PyTorch fp32 参考输出的 token/text 一致 |
| 精度 | fp32 ncnn 路径 |
| Prompt | 内置 `spotting` / `document` 模式，也支持自定义 `--prompt` 文本 |
| Vision | dynamic vision backend，并保留 fixed-grid fallback |

当前已验证配置使用 `max_pixels=524288`。`image_grid_thw` 是 HunyuanOCR 图像预处理后得到的 `[t,h,w]` patch grid。dynamic vision 包使用一份 `vision/vision.ncnn.param/bin` 和 `vision/pos_embed.bin`；fixed-grid 包使用 `vision/grid_38x52/` 这类目录。

当前交付范围不包含原版高分辨率路径。

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
- 已构建或已安装的 ncnn

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

Windows 验证分成两类：`.github/workflows/windows-compile.yml` 在 CI 中只做 MSVC 编译验证；带模型运行是在一台真实 Windows 机器上手动完成。

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

- dynamic vision backend 已在 28 张示例图上完成 token/text 回归验证，验证口径为 `max_pixels=524288`。
- 自定义 prompt 已接入 C++ tokenizer encode；更多 tokenizer 边界还需要继续补充 HF 对齐测试。
- 当前交付范围使用 `max_pixels=524288`，不包含原版高分辨率路径。
- 公开示例脚本只验证端到端运行；严格 token/text 对齐由 fixture 回归验证。

## 许可证

本仓库原创代码使用 Apache-2.0。`third_party/stb_image.h` 保留上游 MIT/Public Domain 许可说明。

HunyuanOCR 模型文件遵循 Tencent Hunyuan Community License Agreement。
