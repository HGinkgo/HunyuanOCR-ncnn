<div align="center">
  <h1>HunyuanOCR-ncnn</h1>
  <p>
    A pure C++17 OCR runtime for
    <a href="https://huggingface.co/tencent/HunyuanOCR"><strong>HunyuanOCR 1.5</strong></a>
    on <a href="https://github.com/Tencent/ncnn"><strong>ncnn</strong></a>
  </p>
  <p>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml/badge.svg" alt="Linux CI"></a>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml/badge.svg" alt="Windows CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
    <img src="https://img.shields.io/badge/C%2B%2B-17-f34b7d.svg" alt="C++17">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg" alt="Linux and Windows">
  </p>
  <p>
    <a href="https://github.com/Tencent/ncnn/discussions/6808">Technical Discussion</a>
    &nbsp;|&nbsp; <a href="README.md">中文说明</a>
  </p>
</div>

---

This project exports the Hugging Face HunyuanOCR model into ncnn submodules with
pnnx and implements image preprocessing, dynamic vision, prompts, KV-cache decoding,
and tokenizer postprocessing in C++.

> `main` is the HunyuanOCR 1.5 preview (`0.4.0`). The `feat/hunyuanocr-1.0` branch
> and `v0.2.0` retain the HunyuanOCR 1.0 version.

## Features

- PNG/JPEG input and dynamic image sizes within the exported processor range.
- Built-in `spotting` and `document` modes plus custom UTF-8 prompts.
- Reusable C++ runtime, per-token streaming callbacks, and JSONL batch inference.
- CPU fp32 by default, with optional DFlash, mmap weight loading, and Vulkan vision.
- Linux and Windows support, including UTF-8 paths and command-line input.
- Strict token/text regression and Sanitizer gates over public images.

## Quick Start

The pre-converted runtime model is hosted on
[ModelScope](https://modelscope.cn/models/HGinkgo/HunyuanOCR-1.5-ncnn):

```bash
python -m pip install modelscope
modelscope download \
  --model HGinkgo/HunyuanOCR-1.5-ncnn \
  --local_dir ./hunyuan_ocr_ncnn_model
```

> This is an unofficial community conversion maintained by an individual. It is not
> released, endorsed, or supported by Tencent.

The script auto-detects `./hunyuan_ocr_ncnn_model` and a sibling `../ncnn/build/src`, then builds and runs an example:

```bash
scripts/quickstart_existing_model.sh
```

Pass `--model PATH` or `--ncnn-dir PATH` when either dependency is elsewhere.
Quickstart generates 16 tokens for a fast smoke test; use the single-image command below for complete OCR.

Model weights are not stored in this source repository.

## Build

Requirements are CMake 3.18, a C++17 compiler, and ncnn. The validated ncnn revision is
`dda2e28bae2a084760361197d87f06e685604e52`.

```bash
cmake -S . -B build -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build build -j
```

<details>
<summary>Use a local ncnn build directory directly</summary>

```bash
cmake -S . -B build \
  -DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF \
  -DNCNN_INCLUDE_DIR=/path/to/ncnn/src \
  -DNCNN_BUILD_INCLUDE_DIR=/path/to/ncnn/build/src \
  -DNCNN_LIBRARY=/path/to/ncnn/build/src/libncnn.a
cmake --build build -j
```

</details>

## Run and Integrate

### Single-image inference

```bash
./build/hunyuan_ocr_cli \
  --image ./examples/images/hf_demo_tools-dark.png
```

The CLI auto-detects `./hunyuan_ocr_ncnn_model` and common `assets/` directories. Single-image inference defaults to the `document` prompt and streams decoded text while it is generated. The default limit is 8192 tokens, with early stopping on EOS or tail repetition. Use `--prompt-mode spotting` when coordinates are needed, or pass a custom prompt with `--prompt "Return only the visible text"`.

### JSONL batch inference

Each line is one request. Specify exactly one of `prompt_mode` and `prompt`:

```json
{"id":"page-1","image":"images/page-1.png","prompt_mode":"document","max_tokens":256}
```

```bash
./build/hunyuan_ocr_cli \
  --batch-input requests.jsonl \
  --batch-output results.jsonl
```

The model loads once and output order matches input order. Failed records contain
`ok: false`; pass `--force` to replace an existing output file.

### C++ runtime

See [`examples/ocr_main.cpp`](examples/ocr_main.cpp) for a minimal runnable integration:

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

One runtime can process requests sequentially. Use `infer_rgb` when the caller already
owns contiguous RGB pixels.

## Optional Capabilities

| Capability | Enable with | Notes |
| --- | --- | --- |
| DFlash | `--dflash` | AR remains the default; low-acceptance inputs may be slower, so no universal speedup is claimed. |
| mmap weights | `--mmap-weights` | Reduces loading copies and anonymous memory; model files remain the same size. |
| Vulkan vision | `--vision-vulkan` | Requires [`patches/ncnn`](patches/ncnn); text generation remains CPU fp32. |

Select a Vulkan device with `--vision-vulkan-device N`. C++ callers can set the
corresponding fields in `RuntimeOptions`.

## More Commands

```bash
# Show CLI options
./build/hunyuan_ocr_cli --help
# Show project and ncnn versions
./build/hunyuan_ocr_cli --version
# Run a quick smoke test with the default example image
scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model
# List bundled examples
python tools/run_example.py --list
# Run one bundled example
python tools/run_example.py --model ./hunyuan_ocr_ncnn_model --case hf_demo
# Run all bundled examples in order
python tools/run_examples.py --model ./hunyuan_ocr_ncnn_model
# Benchmark one example
python tools/benchmark.py --model ./hunyuan_ocr_ncnn_model --cases hf_demo
```

Images and attribution are under [`examples`](examples). Benchmark fields are documented
in [`tools/README.md`](tools/README.md#benchmark).

## Limitations

- HunyuanOCR 1.5 remains a `0.4.0` preview; use the model package linked above.
- The current package uses `max_pixels=524288` and omits the original high-resolution path.
- JPEG decoder rounding can affect decision-sensitive images; use PNG for stable reproduction.
- Custom prompts do not yet cover every tokenizer boundary input.
- Vulkan applies only to the vision encoder; text generation remains CPU fp32.

## License

Original code is Apache-2.0. Third-party licenses for `stb_image.h` and picojson are in
[`NOTICE`](NOTICE). HunyuanOCR model files follow the Tencent Hunyuan Community License Agreement.
