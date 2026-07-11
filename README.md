# HunyuanOCR-ncnn

[![Linux CI](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml)
[![Windows Compile](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml/badge.svg)](https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml)

C++17/ncnn runtime for Tencent HunyuanOCR.

[中文说明](README_zh.md)

Technical report: [Tencent ncnn Discussion #6808](https://github.com/Tencent/ncnn/discussions/6808)

This repository exports the Hugging Face HunyuanOCR model into ncnn submodules
with pnnx and runs the full OCR path in C++.

> **HunyuanOCR 1.5 preview (`0.3.0`):** the current development branch targets
> checkpoint revision `9e01f897bf8956f77a80c350dc0491d6bbbd43e6`. The strict
> reference uses Transformers 5.13.0 with CPU fp32 eager attention. Linux and
> Windows validation passed, and all 28 bundled cases match for the first 128
> generated tokens and decoded text. Tag `v0.2.0` remains the frozen
> HunyuanOCR 1.0 release.

| Line | Model | Status |
| --- | --- | --- |
| `main` | HunyuanOCR 1.5 | `0.3.0` development/preview |
| `feat/hunyuanocr-1.0` | HunyuanOCR 1.0 | preserved compatibility branch |
| `v0.2.0` | HunyuanOCR 1.0 | frozen release |

## Highlights

- End-to-end PNG/JPEG input to OCR text in C++17.
- One dynamic vision package for exported image sizes, with fixed-grid fallback.
- KV-cache text decoder, greedy decoding, and repetition penalty.
- Built-in `spotting` / `document` prompts plus custom `--prompt` text.
- CMake build on Linux and Windows; no Python at runtime.
- 28 bundled cases match the HunyuanOCR 1.5 PyTorch fp32 reference for the
  validated 128-token window.

## Quick Links

| Need | Entry |
| --- | --- |
| Build and run one image | `scripts/quickstart_existing_model.sh` |
| Package converted artifacts | `tools/package_model.py` |
| Export from HF weights | `export/README.md` |
| Run examples | `tools/run_example.py`, `tools/run_examples.py` |
| Run strict regression | `tools/run_regression.py` |
| Benchmark runtime | `tools/benchmark.py`, `benchmark/README.md` |
| Model layout | `models/README.md`, `models/model.json.example` |

## Status

| Item | Status |
| --- | --- |
| Linux | HunyuanOCR 1.5 build, CTest, and 28-case 128-token regression validated locally |
| Windows | Build and packaged-model validation passed |
| Runtime | PNG/JPEG input to final OCR text |
| Validation | 28 bundled cases match the 1.5 PyTorch fp32 reference token/text for 128 generated tokens |
| Precision | fp32 ncnn path |
| Prompts | built-in `spotting`/`document` modes and custom `--prompt` text |
| Vision | one dynamic package for exported image sizes, with fixed-grid fallback |

The current verified configuration uses `max_pixels=524288`. In practice, the
dynamic vision package covers image sizes inside the exported processor range,
and fixed-grid packages are kept as a compatibility fallback. See
`models/README.md` for package layout details.

This version does not cover the original high-resolution route.

PNG is used as the canonical strict input for JPEG cases that are sensitive to
decoder rounding. JPEG remains supported for normal inference, but Pillow/
libjpeg-turbo and `stb_image` may decode lossy JPEG pixels slightly differently;
cross-decoder token identity is therefore not guaranteed for every JPEG.

## Quick Start

If you already have a packaged `hunyuan_ocr_ncnn_model/` directory, the shortest
path is:

```bash
scripts/quickstart_existing_model.sh \
  --model ./hunyuan_ocr_ncnn_model \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn
```

To smoke-test one image after building:

```bash
scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model
```

### 1. Build

Requirements:

- CMake 3.18 or newer
- C++17 compiler
- ncnn built or installed

With an installed ncnn CMake package:

```bash
cmake -S . -B build -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build build -j
```

With direct paths to a local ncnn build:

```bash
cmake -S . -B build \
  -DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF \
  -DNCNN_INCLUDE_DIR=/path/to/ncnn/src \
  -DNCNN_BUILD_INCLUDE_DIR=/path/to/ncnn/build/src \
  -DNCNN_LIBRARY=/path/to/ncnn/build/src/libncnn.a
cmake --build build -j
```

Basic checks:

```bash
./build/hunyuan_ocr_cli --help
./build/hunyuan_ocr_cli --version
```

Windows build and packaged-model validation passed.

### 2. Package Model Files

`tools/package_model.py` creates the runtime model directory from exported
artifacts. Here `<workspace>` means the directory that contains
`models/tokenizer/` and `models/export/`.

```bash
python tools/package_model.py \
  --workspace <workspace> \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

Use symlinks instead of copies by omitting `--copy`. Use
`--vision-backend fixed` for the v0.1 fixed-grid package, or `both` to include
dynamic vision and fixed-grid fallback files in one model directory.

### 3. Run Examples

List bundled images:

```bash
python tools/run_example.py --list
```

Run one image:

```bash
python tools/run_example.py \
  --model ./hunyuan_ocr_ncnn_model \
  --case hf_demo
```

Run one bundled image with a custom prompt:

```bash
python tools/run_example.py \
  --model ./hunyuan_ocr_ncnn_model \
  --case hf_demo \
  --prompt "Describe the visible UI text."
```

Run all bundled images:

```bash
python tools/run_examples.py \
  --model ./hunyuan_ocr_ncnn_model
```

Dynamic vision packages support different image sizes within the exported
processor range. Fixed-grid fallback packages use the `grid_<h>x<w>` naming
convention, for example `vision/grid_38x52/`.

## Model Package

Expected runtime layout:

```text
hunyuan_ocr_ncnn_model/
  model.json
  tokenizer/
  text_embed/
  text_decoder/
  lm_head/
  vision/
    vision.ncnn.param
    vision.ncnn.bin
    pos_embed.bin
    grid_<grid_h>x<grid_w>/
```

The runtime model package is generated separately from converted artifacts. The
repository itself tracks source code, scripts, metadata, and example images.
See `models/README.md` for the `model.json` schema and dynamic/fixed vision
backend selection.

## Export From HF Model

The export scripts live in `export/`. They write tokenizer and ncnn submodule
artifacts into a workspace layout consumed by `tools/package_model.py`.

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

For a Linux end-to-end helper:

```bash
scripts/export_and_package_linux.sh \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn \
  --output ./hunyuan_ocr_ncnn_model \
  --copy
```

See `export/README.md` for the module-level commands.

## Regression

For full token/text regression after preparing fixtures:

```bash
python tools/run_regression.py \
  --package \
  --package-vision-backend dynamic
```

Expected summary:

```text
summary: 28/28 passed
```

This regression compares prompt ids, position ids, generated token ids, and
decoded text.

## Benchmark

After building the CLI and preparing a packaged model:

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases hf_demo \
  --repeat 3 \
  --warmup 1 \
  --max-tokens 64
```

The benchmark separates cold start from same-process warm inference, supports
CPU thread sweeps, and reports stage timing plus decode throughput. See
`benchmark/README.md`.

## Repository Layout

```text
include/hunyuan_ocr/   Public C++ headers
src/                   Runtime and CLI implementation
third_party/           Vendored single-header dependencies
export/                Export workflow notes
scripts/               Quickstart and export/package shell helpers
tools/                 Model packaging and regression helpers
examples/              Example images and usage notes
benchmark/             Runtime benchmark notes
models/                Tracked config template only
```

## Limitations

- The dynamic vision backend is verified on the bundled 28-image regression set
  under `max_pixels=524288` for the first 128 generated tokens.
- HunyuanOCR 1.5 remains a preview tied to the pinned checkpoint revision and
  the validated 128-token window.
- Lossy JPEG decoder rounding may change generated tokens in decision-sensitive
  cases; canonical lossless inputs are used where strict parity requires them.
- Custom prompt text is supported through the C++ tokenizer encode path; broader
  tokenizer edge cases still need more HF parity tests.
- The current delivery scope uses `max_pixels=524288`; it does not include the
  original high-resolution route.
- The public example runner checks runtime execution. Fixture regression is
  used for token/text equality.

## License

Original code in this repository is licensed under Apache-2.0. See `LICENSE`
and `NOTICE`. Vendored `stb_image.h` keeps its upstream MIT/Public Domain
license notice.

HunyuanOCR model files are governed by the Tencent Hunyuan Community License
Agreement.
