# HunyuanOCR-ncnn

C++/ncnn deployment scaffold for Tencent HunyuanOCR.

This repository converts the HunyuanOCR inference path into ncnn submodules and
connects them with a small C++ runtime. The current validated path targets a
deterministic fp32 deployment baseline and compares final generated tokens/text
against a PyTorch fp32 golden run.

Model weights and converted ncnn artifacts are not redistributed in this
repository.

## Current Status

- Linux CMake build passes with a local ncnn build.
- Windows compile-only passes in GitHub Actions with MSVC 2022.
- The current fp32 `max_pixels=524288` route matches the PyTorch fp32 golden
  output on the five local regression images.
- The CLI can run PNG/JPEG input through C++ image decode, PIL-compatible
  resize, patch flattening, fixed-grid ncnn vision, built-in prompt assembly,
  text decoding, and tokenizer decode.
- Supported prompt modes are currently `spotting` and `document`.

The implementation is intentionally conservative: fixed-grid vision packages
are used first so model conversion, runtime wiring, and token alignment can be
verified before general dynamic-grid packaging is added.

## Supported Baseline

| Item | Current value |
| --- | --- |
| Precision | fp32 ncnn runtime path |
| Image processor bounds | `min_pixels=262144`, `max_pixels=524288` |
| Vision input boundary | flattened patches `[patch_count, 768]` |
| Vision packages | fixed grid directories under `vision/grid_<h>x<w>/` |
| Current regression grids | `38x52`, `54x36`, `58x34` |
| Text decoder | KV cache greedy decode |
| Repetition penalty | `1.03` |
| EOS ids | `120007`, `120020` |
| Prompts | built-in `spotting` and `document` templates |

The default high-resolution HunyuanOCR route and arbitrary user prompt encoding
are not part of the current validated package.

## Build

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

Basic CLI checks:

```bash
./build/hunyuan_ocr_cli --help
./build/hunyuan_ocr_cli --version
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model
```

Windows compile-only is covered by `.github/workflows/windows-compile.yml`.
The workflow builds ncnn with MSVC 2022, installs the ncnn SDK into the Actions
workspace, configures this project through `ncnn_DIR`, and builds the CLI. It
does not download models or run OCR inference.

## Model Packaging

`tools/package_model.py` creates the standard runtime model directory from
exported artifacts in the outer project workspace:

```bash
python tools/package_model.py \
  --workspace /root/hpf/workspace/ncnn_hunyuanocr \
  --output /tmp/hunyuanocr_ncnn_model_packaged \
  --force
```

The default mode creates symlinks. Use `--copy` when creating a portable bundle
or when symlink creation is unavailable:

```bash
python tools/package_model.py \
  --workspace /root/hpf/workspace/ncnn_hunyuanocr \
  --output /tmp/hunyuanocr_ncnn_model_packaged \
  --copy \
  --force
```

Packaged layout:

```text
hunyuan_ocr_ncnn_model/
  model.json
  tokenizer/
    vocab.txt
    merges.txt
    special_tokens.json
    eos_ids.json
  text_embed/
    text_embed.ncnn.param
    text_embed.ncnn.bin
  text_decoder/
    text_decoder_kv.ncnn.param
    text_decoder_kv.ncnn.bin
  lm_head/
    lm_head.ncnn.param
    lm_head.ncnn.bin
  vision/
    grid_<grid_h>x<grid_w>/
      vision.ncnn.param
      vision.ncnn.bin
```

The package contains only runtime files. PyTorch checkpoints, `.pt`, `.npy`,
`.npz`, conversion logs, and intermediate export artifacts stay outside the
repository and outside the packaged runtime directory.

## Run

Run the current raw-image path with a packaged model:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --image /root/hpf/workspace/ncnn_hunyuanocr/datasets/test_images/hf_demo_tools-dark.png \
  --prompt-mode spotting
```

For document parsing style output:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --image /path/to/document.png \
  --prompt-mode document
```

The image must preprocess to a grid that exists in the packaged model directory.
If no matching `vision/grid_<grid_h>x<grid_w>/` package exists, the CLI reports
the missing grid explicitly.

## Regression

The local five-sample regression can rebuild the packaged model and run all
current golden cases with one command:

```bash
python tools/run_5sample_regression.py --package
```

Expected summary:

```text
summary: 5/5 passed
```

Per-case logs are written to `/tmp/hunyuanocr_5sample_regression/`.

## Conversion Layout

The runtime is assembled from these pnnx/ncnn conversion outputs:

| Component | Runtime path | Notes |
| --- | --- | --- |
| tokenizer | `tokenizer/` | decode path implemented in C++; encode for arbitrary prompt is pending |
| text embedding | `text_embed/text_embed.ncnn.*` | token id to hidden embedding |
| text decoder | `text_decoder/text_decoder_kv.ncnn.*` | external RoPE and KV cache path |
| lm head | `lm_head/lm_head.ncnn.*` | tied embedding weight exported explicitly |
| vision | `vision/grid_<h>x<w>/vision.ncnn.*` | fixed-grid flattened patch input |

The validated conversion baseline is fp32. ncnn fp16/bf16 runtime options are
not used for the current correctness target.

## Known Limitations

- Current vision packaging is fixed-grid. Images whose preprocessed grid is not
  packaged will not run until that grid is exported and added.
- Arbitrary user prompts are not yet supported because general C++ BPE
  encode/chat-template assembly is still pending. Use `--prompt-mode spotting`
  or `--prompt-mode document`.
- The default high-resolution HunyuanOCR route is not part of the current
  package; the validated route uses `max_pixels=524288`.
- Windows has compile-only coverage. Loading the model and running OCR on a
  Windows machine is a separate validation step.
- HunyuanOCR model files are governed by the Tencent Hunyuan Community License
  Agreement and are not included here.

## Repository Layout

```text
include/hunyuan_ocr/   Public C++ headers
src/                   Runtime and CLI implementation
third_party/           Vendored single-header dependencies such as stb_image
export/                Export script landing area
tools/                 Model packaging and regression helpers
examples/              Usage examples and development validation notes
models/                Only example config is tracked; real models are ignored
```

## License

Original code in this repository is licensed under Apache-2.0. See `LICENSE`
and `NOTICE`. Vendored `stb_image.h` keeps its upstream MIT/Public Domain
license notice.

HunyuanOCR model files are governed by the Tencent Hunyuan Community License
Agreement and are not redistributed here.
