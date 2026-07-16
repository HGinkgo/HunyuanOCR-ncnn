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
    &nbsp;|&nbsp; <a href="README_zh.md">中文说明</a>
  </p>
</div>

---

This repository exports the Hugging Face HunyuanOCR model into ncnn submodules
with pnnx and runs the full OCR path in C++.

> **HunyuanOCR 1.5 preview (`0.4.0`):** the current development branch targets
> checkpoint revision `9e01f897bf8956f77a80c350dc0491d6bbbd43e6`. The strict
> reference uses Transformers 5.13.0 with CPU fp32 eager attention. Linux and
> Windows validation passed, including the 28-case test suite. Tag `v0.2.0`
> remains the frozen HunyuanOCR 1.0 release.

| Line | Model | Status |
| --- | --- | --- |
| `main` | HunyuanOCR 1.5 | `0.4.0` development/preview |
| `feat/hunyuanocr-1.0` | HunyuanOCR 1.0 | preserved compatibility branch |
| `v0.2.0` | HunyuanOCR 1.0 | frozen release |

## Competition Coverage

| Requirement | Evidence in this repository |
| --- | --- |
| Convert HunyuanOCR with pnnx | Reproducible module exports under `export/` and packaging through `tools/package_model.py` |
| C++ LLM decoding with few dependencies | The runtime and KV-cache decoder use [ncnn_llm](https://github.com/nihui/ncnn_llm) as an architecture reference; the C++17 executable uses ncnn plus the bundled `stb_image` and `picojson` |
| Match the PyTorch final text | The pinned Transformers 5.13.0 CPU fp32 reference and ncnn runtime pass all 28 token/text cases for the verified 128-token window |
| CMake on at least two platforms | Linux and Windows build, test, and packaged-model validation are covered by CI |
| Publish a technical summary | [Tencent/ncnn Discussion #6808](https://github.com/Tencent/ncnn/discussions/6808) links back to this repository |

The HunyuanOCR 1.5 adapter and its validation are maintained in this repository;
the ncnn_llm link above records the required reference project rather than
claiming that this repository is an upstream ncnn_llm branch.

## Advanced Engineering

- One dynamic vision package for exported image sizes, with fixed-grid fallback.
- Append-only, capacity-bearing KV caches avoid steady-state past-cache copies;
  dedicated lifecycle tests cover growth, logical views, and repeated requests.
- Built-in `spotting` / `document` prompts plus custom UTF-8 `--prompt` text.
- A reusable C++ runtime keeps model networks loaded across sequential requests;
  strict JSONL batch processing preserves input order and reports per-record errors.
- Windows CLI supports UTF-8 prompts and Unicode model, image, and fixture paths.
- Optional DFlash speculative decoding with AR kept as the default path.
- Optional fp32 Vulkan vision backend; the 28-case token/text test suite passes
  without GELU CPU fallback when using the maintained ncnn patch series.
- A 100-round RSS regression and ASAN/UBSAN test gate audit request-scoped
  ncnn buffer release for long-lived vision, text, and DFlash runtimes.

## Quick Links

| Need | Entry |
| --- | --- |
| Build and run one image | `scripts/quickstart_existing_model.sh` |
| Package converted artifacts | `tools/package_model.py` |
| Export from HF weights | `export/README.md` |
| Run examples | `tools/run_example.py`, `tools/run_examples.py` |
| Run strict regression | `tools/run_regression.py` |
| Benchmark runtime | `tools/benchmark.py`, `tools/README.md#benchmark` |
| Model layout | `models/README.md`, `models/model.json.example` |

The current verified configuration uses `max_pixels=524288` because development
and validation were limited to a single RTX 3090 (24 GB), requiring bounded GPU
memory use during model conversion and testing. This is the project's verified
scope, not a theoretical limit of HunyuanOCR or ncnn. In practice, the dynamic
vision package covers image sizes inside the exported processor range, and
fixed-grid packages are kept as a compatibility fallback. See `models/README.md`
for package layout details.

This version does not cover the original high-resolution route.

PNG is used as the canonical strict input for JPEG cases that are sensitive to
decoder rounding. JPEG remains supported for normal inference, but Pillow/
libjpeg-turbo and `stb_image` may decode lossy JPEG pixels slightly differently;
cross-decoder token identity is therefore not guaranteed for every JPEG.

## Experimental DFlash

`--dflash` enables the optional DFlash speculative decoder for greedy generation.
It requires a package containing `dflash/dflash.ncnn.param/bin` and the auxiliary
text decoder exported with `out1` through `out4`. AR remains the default and the
runtime does not switch decoding methods automatically.

Linux and Windows validation passed with token/text output identical to AR. The
performance gain is input-dependent: in the current three-case CPU benchmark,
warm speedup ranged from `0.64x` to `1.20x` as draft acceptance increased from
`2.71%` to `17.04%`. DFlash may be slower on low-acceptance inputs, so it remains
an explicit development/preview option rather than a general acceleration claim.

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode document \
  --dflash
```

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

The repository does not include model weights. If you already have a runtime
model package, use the commands above. To reproduce the conversion from the
Hugging Face model, follow the export, package, build, and run sections below.

## Export From HF Model

The scripts in `export/` use pnnx to convert the Hugging Face model into
tokenizer files and ncnn submodule artifacts:

```bash
python export/export_all.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --workspace .
```

See `export/README.md` for the module-level commands.

## Package the ncnn Runtime Model

`tools/package_model.py` arranges exported artifacts into the runtime model
package consumed by the CLI:

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

```bash
python tools/package_model.py \
  --workspace <workspace> \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

Add `--dflash` to package the draft network and auxiliary decoder from their
default export directories. `--dflash-dir`, `--dflash-decoder-dir`, and
`--base-runtime-dir` allow those inputs to be supplied explicitly.

Here `<workspace>` means the directory containing `models/tokenizer/` and
`models/export/`. Use symlinks instead of copies by omitting `--copy`. Use
`--vision-backend fixed` for the v0.1 fixed-grid package, or `both` to include
dynamic vision and fixed-grid fallback files in one model directory. See
`models/README.md` for the `model.json` schema and backend selection rules.

On Linux, the helper below combines build, export, packaging, and an example run:

```bash
scripts/export_and_package_linux.sh \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn \
  --output ./hunyuan_ocr_ncnn_model \
  --copy
```

## Build

Requirements:

- CMake 3.18 or newer
- C++17 compiler
- ncnn `20260106` or newer; validation pins revision
  `244f30c8b995d5b2cf57b59950596490c68813d6`

The default CPU build works with the unmodified pinned ncnn checkout. The
optional fp32 Vulkan vision backend uses the project-maintained patch series in
[`patches/ncnn`](patches/ncnn):

```bash
python scripts/apply_ncnn_patches.py --ncnn-dir /path/to/ncnn
```

The Vulkan MatMul implementation is derived from
[Cat-myq's proposed Tencent/ncnn PR #6579](https://github.com/Tencent/ncnn/pull/6579)
at commit `88e0927f6e6b640fea19bd5721ff5409fcca99ef`; it is not described here as
merged upstream. The second patch adds the exact fp32 GELU path required by the
HunyuanOCR vision encoder. Source attribution and the ncnn BSD 3-Clause license
are retained in the patch bundle and `NOTICE`.

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

Windows build and packaged-model validation passed. The CLI reads the native
wide-character command line and uses UTF-8 for prompts, console I/O, and model,
image, and fixture paths.

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

Dynamic vision packages support different image sizes within the exported
processor range. Fixed-grid fallback packages use the `grid_<h>x<w>` naming
convention, for example `vision/grid_38x52/`.

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
license notice, and vendored picojson keeps its BSD 2-Clause license.

HunyuanOCR model files are governed by the Tencent Hunyuan Community License
Agreement.
