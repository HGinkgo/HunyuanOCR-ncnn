# HunyuanOCR-ncnn

C++/ncnn runtime for Tencent HunyuanOCR.

[中文说明](README_zh.md)

Technical report: [Tencent ncnn Discussion #6808](https://github.com/Tencent/ncnn/discussions/6808)

This project converts the HunyuanOCR inference path into ncnn modules with
pnnx, then connects image preprocessing, dynamic/fixed-grid vision inference, KV-cache
text decoding, lm head, and tokenizer decode in C++.

## Status

| Item | Status |
| --- | --- |
| Linux | CMake build and 10-image runtime regression validated locally |
| Windows CI | MSVC compile-only in GitHub Actions |
| Windows runtime | Packaged-model run validated on a real Windows machine |
| Runtime | PNG/JPEG input to final OCR text |
| Validation | 10 bundled regression images match PyTorch fp32 reference token/text |
| Precision | fp32 ncnn path |
| Prompts | built-in `spotting`/`document` modes and custom `--prompt` text |
| Vision | dynamic vision backend, with fixed-grid fallback |

The current verified configuration uses `max_pixels=524288`. `image_grid_thw`
is the `[t,h,w]` patch grid produced by the HunyuanOCR image processor. The
dynamic vision package uses one `vision/vision.ncnn.param/bin` pair plus
`vision/pos_embed.bin`; fixed-grid packages use directories such as
`vision/grid_38x52/`.

The current delivery scope does not include the original high-resolution route.

## Quick Start

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

Windows coverage is split into two checks:
`.github/workflows/windows-compile.yml` runs MSVC compile-only in CI, and a
separate real Windows machine was used for packaged-model runtime validation.

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

The image must preprocess to a supported grid. Runtime vision directories use
the `grid_<h>x<w>` naming convention, for example `image_grid_thw=[1,38,52]`
uses `vision/grid_38x52/`.

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

The repository tracks source code, scripts, example images, and metadata. The
runtime model package is generated separately from converted artifacts.

## Regression

For full token/text regression after preparing fixtures:

```bash
python tools/run_regression.py \
  --package \
  --package-vision-backend dynamic
```

Expected summary:

```text
summary: 10/10 passed
```

This regression compares prompt ids, position ids, generated token ids, and
decoded text.

## Repository Layout

```text
include/hunyuan_ocr/   Public C++ headers
src/                   Runtime and CLI implementation
third_party/           Vendored single-header dependencies
export/                Export workflow notes
tools/                 Model packaging and regression helpers
examples/              Example images and usage notes
models/                Tracked config template only
```

## Limitations

- The dynamic vision backend is verified on the bundled 10-image regression set
  under `max_pixels=524288`.
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
