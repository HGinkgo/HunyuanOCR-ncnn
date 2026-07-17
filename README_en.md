<div align="center">
  <h1>HunyuanOCR-ncnn</h1>
  <p>
    <a href="https://huggingface.co/tencent/HunyuanOCR"><strong>HunyuanOCR 1.5</strong></a>
    on <a href="https://github.com/Tencent/ncnn"><strong>ncnn</strong></a> -
    a pure C++17 OCR inference runtime
  </p>
  <p>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml/badge.svg" alt="Linux CI"></a>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml/badge.svg" alt="Windows CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
    <img src="https://img.shields.io/badge/C%2B%2B-17-f34b7d.svg" alt="C++17">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg" alt="Linux and Windows">
    <img src="https://img.shields.io/badge/backend-CPU%20fp32%20%7C%20Vulkan%20vision%20fp32-4c1.svg" alt="CPU fp32 and Vulkan vision fp32">
  </p>
  <p>
    Tencent/ncnn activity entry <a href="https://github.com/Tencent/ncnn/discussions/6808">#6808</a>
    &nbsp;|&nbsp; <a href="README.md">中文说明</a>
  </p>
</div>

---

This repository exports the Hugging Face HunyuanOCR model into ncnn submodules
with pnnx and runs the full OCR path in C++.

> **HunyuanOCR 1.5 preview (`0.4.0`):** `main` is paired with the ModelScope
> package below. Tag `v0.2.0` remains the frozen HunyuanOCR 1.0 release.

| Line | Model | Status |
| --- | --- | --- |
| `main` | HunyuanOCR 1.5 | `0.4.0` development/preview |
| `feat/hunyuanocr-1.0` | HunyuanOCR 1.0 | preserved compatibility branch |
| `v0.2.0` | HunyuanOCR 1.0 | frozen release |

## Quick Start

The pre-converted ncnn runtime model is hosted separately on
[ModelScope](https://modelscope.cn/models/HGinkgo/HunyuanOCR-1.5-ncnn):

```bash
python -m pip install modelscope
modelscope download \
  --model HGinkgo/HunyuanOCR-1.5-ncnn \
  --local_dir ./hunyuan_ocr_ncnn_model
```

> This model package is an unofficial community artifact converted and
> maintained by an individual. It is not an official Tencent release and is
> not endorsed by Tencent.

After downloading the model, build the project and run one example with:

```bash
scripts/quickstart_existing_model.sh \
  --model ./hunyuan_ocr_ncnn_model \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn
```

To smoke-test one image after building:

```bash
scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model
```

The source repository does not embed model weights. Download the runtime package
above to get started.

## Quick Links

| Need | Entry |
| --- | --- |
| Download the pre-converted model | [ModelScope](https://modelscope.cn/models/HGinkgo/HunyuanOCR-1.5-ncnn) |
| Build and run one image | `scripts/quickstart_existing_model.sh` |
| Run examples | `tools/run_example.py`, `tools/run_examples.py` |
| JSONL batch inference | `--batch-input`, `--batch-output` |
| C++ integration | `hunyuan_ocr` static library |
| Benchmark runtime | `tools/benchmark.py`, `tools/README.md#benchmark` |

## Features

- PNG/JPEG input and dynamic image sizes within the exported range.
- Built-in `spotting` / `document` modes plus custom UTF-8 `--prompt` text.
- Reusable C++ runtime for processing multiple images without reloading models.
- Ordered JSONL batch inference with a separate error result for each failed row.
- UTF-8 prompts, model paths, and image paths on both Linux and Windows.
- Optional DFlash speculative decoding, read-only mmap weights, and fp32 Vulkan vision.

## DFlash

`--dflash` enables the optional DFlash speculative decoder for greedy generation;
the ModelScope package includes the required files. AR remains the default and
the runtime does not switch decoding methods automatically.

Performance depends on the input. DFlash may be slower on low-acceptance inputs,
so it is not presented as a general acceleration option.

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode document \
  --dflash
```

## Optional mmap Weight Loading

`--mmap-weights` loads the vision, text, and DFlash ncnn weights from read-only
file mappings. The existing loader remains the default. This option reduces
model-loading copies and anonymous memory use.

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode document \
  --mmap-weights
```

C++ callers can set `RuntimeOptions::mmap_weights = true`. This option does not
shrink model files or change inference numerics.

## Build

Requirements:

- CMake 3.18 or newer
- C++17 compiler
- ncnn `20260106` or newer; validation pins revision
  `dda2e28bae2a084760361197d87f06e685604e52`

The default CPU build works with the unmodified pinned ncnn checkout. The
optional fp32 Vulkan vision backend uses the project-maintained patch series in
[`patches/ncnn`](patches/ncnn):

```bash
python scripts/apply_ncnn_patches.py --ncnn-dir /path/to/ncnn
```

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

Run the vision encoder with Vulkan after building the patched ncnn SDK:

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode spotting \
  --vision-vulkan \
  --vision-vulkan-device 0
```

Only the vision encoder uses Vulkan; text generation continues to use the
existing CPU fp32 path.

The Windows CLI supports Unicode prompts, model paths, and image paths, and uses
UTF-8 for console input and output.

## Reusable C++ Runtime

The `hunyuan_ocr` static target owns the loaded model and can process multiple
requests without reloading its ncnn networks. Add the source tree and link it:

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

`infer_rgb` accepts a contiguous RGB byte vector when the caller already owns
decoded pixels. One runtime processes requests sequentially and is not safe for
concurrent calls; create independent instances when duplicated model memory is
acceptable. Request-local image tensors, features, and KV caches are released
after each call.

## JSONL Batch Inference

Batch mode loads the model once and writes one ordered result for every physical
input line:

```json
{"id":"page-1","image":"images/page-1.png","prompt_mode":"document","max_tokens":256}
{"id":"page-2","image":"images/page-2.png","prompt":"Only return the invoice number."}
```

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --batch-input requests.jsonl \
  --batch-output results.jsonl \
  --num-threads 8
```

`id` must be non-empty and unique. Each record contains exactly one of
`prompt_mode` (`spotting` or `document`) and `prompt`; `max_tokens` is optional.
Relative image paths are resolved from the input JSONL directory. Invalid or
failed records are written with `ok: false` and processing continues, while the
final process exit code is nonzero if any record failed. Existing output files
are rejected unless `--force` is supplied.

## Run Examples

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

The canonical dynamic vision network supports different image sizes within
the exported processor range.

## Benchmark

After building the CLI and downloading the model:

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
`tools/README.md#benchmark`.

## Repository Layout

```text
include/hunyuan_ocr/   Public C++ headers
src/                   Runtime and CLI implementation
third_party/           Vendored single-header dependencies
export/                Export workflow notes
scripts/               Quickstart and export/package shell helpers
tools/                 Model packaging and regression helpers
examples/              Example images and usage notes
models/                Tracked config template only
```

## Limitations

- HunyuanOCR 1.5 remains a `0.4.0` preview; use the model package linked in this README.
- The current package uses `max_pixels=524288` and does not include the original
  high-resolution route.
- JPEG decoder rounding can affect a small number of decision-sensitive images;
  use PNG when stable reproduction matters.
- Custom UTF-8 prompts are supported, but not every tokenizer edge input has
  been covered.
- Vulkan is limited to the vision encoder; text generation remains on CPU fp32.

## License

Original code in this repository is licensed under Apache-2.0. See `LICENSE`
and `NOTICE`. Vendored `stb_image.h` keeps its upstream MIT/Public Domain
license notice, and vendored picojson keeps its BSD 2-Clause license.

HunyuanOCR model files are governed by the Tencent Hunyuan Community License
Agreement.
