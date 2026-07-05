# HunyuanOCR-ncnn

C++/ncnn deployment workspace for Tencent HunyuanOCR.

Current status: Phase 5 fixture pipeline. The repository can build a C++
CLI linked with ncnn, validate the expected model directory layout, decode
generated token ids with the exported tokenizer vocabulary, run
`text_embed + text_decoder + lm_head`, inject vision features, and validate a
fixed-grid `PNG/JPEG image -> RGB -> PIL-compatible resize -> pixel_values ->
vision_features -> built-in prompt -> decoder` development path. It does not
yet provide arbitrary text prompts because generic C++ BPE encode/chat-template
support and dynamic vision packaging are still pending.

## Scope

- Convert HunyuanOCR submodules with pnnx and run them with ncnn.
- Keep model weights and converted artifacts outside git.
- Target a deterministic fp32 deployment path first.
- Match the PyTorch fp32 baseline text output for the same images and prompts.
- Build with CMake on Linux and Windows.

The current validated baseline is the fp32 capped route with
`max_pixels=524288`. The default high-resolution route remains a later
extension point.

## Build

Use either an installed ncnn package or direct paths to a local ncnn build.

Example with direct paths:

```bash
cmake -S . -B build \
  -DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF \
  -DNCNN_INCLUDE_DIR=/path/to/ncnn/src \
  -DNCNN_BUILD_INCLUDE_DIR=/path/to/ncnn/build/src \
  -DNCNN_LIBRARY=/path/to/ncnn/build/src/libncnn.a
cmake --build build -j
```

Windows compile-only is covered by `.github/workflows/windows-compile.yml`.
The workflow builds ncnn with MSVC 2022, installs the ncnn SDK into the Actions
workspace, then configures this project through `ncnn_DIR` and builds the CLI.
It does not download models or run OCR inference.

Then run:

```bash
./build/hunyuan_ocr_cli --help
./build/hunyuan_ocr_cli --version
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model --smoke-text 0
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model --decode-ids-file generated_ids.txt
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model --text-fixture /tmp/text_fixture --max-tokens 16
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model --vlm-fixture /tmp/vlm_fixture
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model \
  --vision-param /path/to/vision_fixed_grid.ncnn.param \
  --vision-bin /path/to/vision_fixed_grid.ncnn.bin \
  --vision-fixture /tmp/vision_fixture \
  --vlm-fixture /tmp/vlm_fixture
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model \
  --image-preprocess-fixture /tmp/image_preprocess_fixture \
  --vision-param /path/to/vision_fixed_grid.ncnn.param \
  --vision-bin /path/to/vision_fixed_grid.ncnn.bin \
  --vlm-fixture /tmp/vlm_fixture
./build/hunyuan_ocr_cli --model /path/to/hunyuan_ocr_ncnn_model \
  --image /path/to/image.png \
  --prompt-mode spotting \
  --vlm-fixture /tmp/vlm_fixture
```

## Model Layout

Model files are intentionally ignored by git. A packaged model directory should
follow the layout described by `models/model.json.example`:

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

The current deployment path uses fixed-grid fp32 p512k vision packages. The CLI
selects `vision/grid_<grid_h>x<grid_w>/vision.ncnn.param` and `.bin` after image
preprocessing computes `image_grid_thw`. You can still override this selection
with `--vision-param` and `--vision-bin` for diagnostics.

Use `tools/package_model.py` to build that directory from the exported artifacts
in the outer project workspace. The default mode creates symlinks, so the
packaged directory is cheap to recreate and should still stay outside git:

```bash
python tools/package_model.py \
  --workspace /root/hpf/workspace/ncnn_hunyuanocr \
  --output /tmp/hunyuanocr_ncnn_model_packaged \
  --force
```

Use `--copy` instead of symlinks when creating a portable bundle or when
symlink creation is unavailable:

```bash
python tools/package_model.py \
  --workspace /root/hpf/workspace/ncnn_hunyuanocr \
  --output /tmp/hunyuanocr_ncnn_model_packaged \
  --copy \
  --force
```

The generated directory contains only runtime files: tokenizer assets,
`text_embed`, `text_decoder`, `lm_head`, fixed-grid vision packages, and
`model.json`. It intentionally excludes PyTorch checkpoints, `.pt`, `.npy`,
`.npz`, logs, and conversion scripts.

After packaging, the raw-image path can run without explicit vision network
arguments:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --image /root/hpf/workspace/ncnn_hunyuanocr/datasets/test_images/hf_demo_tools-dark.png \
  --prompt-mode spotting
```

The local five-sample regression can package the model and run all current
golden cases with one command:

```bash
python tools/run_5sample_regression.py --package
```

For local development, the packaged model directory normally points to the
validated artifacts under the project workspace:

```text
tokenizer -> /root/hpf/workspace/ncnn_hunyuanocr/models/tokenizer
text_embed -> /root/hpf/workspace/ncnn_hunyuanocr/models/export/text_embed
text_decoder -> /root/hpf/workspace/ncnn_hunyuanocr/models/export/text_decoder
lm_head -> /root/hpf/workspace/ncnn_hunyuanocr/models/export/lm_head
```

## Text Fixture Format

`--text-fixture` is a development smoke-test path. It expects a directory with
raw little-endian tensors:

```text
meta.txt              # seq_len=<N>, expected_token_count=<M>
inputs_embeds.f32     # float32 [seq_len, 1024], image features already injected
input_ids.i32         # int32 [seq_len]
position_ids.i32      # int32 [4, seq_len]
expected_tokens.i32   # int32 [expected_token_count]
```

This keeps the C++ runtime free of `.npz`/zip dependencies while still allowing
the Phase 4 Python baseline artifacts to be converted into deterministic smoke
fixtures.

`--vlm-fixture` is one step closer to the final runtime. It lets C++ run
`input_ids -> text_embed -> inject vision_features -> decoder loop -> decode
text` while image preprocessing and vision ncnn inference are still external:

```text
meta.txt              # seq_len=<N>, expected_token_count=<M>, image_token_id=<ID>, vision_token_count=<K>
input_ids.i32         # int32 [seq_len]
position_ids.i32      # int32 [4, seq_len]
vision_features.f32   # float32 [vision_token_count, 1024]
expected_tokens.i32   # int32 [expected_token_count]
expected_text.txt     # optional text oracle for detokenized output
```

`--vision-fixture` validates the current fixed-grid vision artifacts. It runs
`pixel_values -> vision_features` in ncnn. When combined with `--vlm-fixture`,
the generated vision features are injected into the text path instead of using
`vision_features.f32` from the VLM fixture:

```text
meta.txt                         # patch_count=<N>, vision_token_count=<K>
pixel_values.f32                 # float32 [patch_count, 768]
expected_vision_features.f32     # optional float32 [vision_token_count, 1024]
```

This is still a development fixture path. Prompt assembly and dynamic-grid
vision packaging remain to be wired before the repository can expose a general
image OCR CLI.

`--image-preprocess-fixture` validates the C++ preprocessing path from resized
RGB bytes to HunyuanOCR flattened patches. When combined with `--vision-param`,
`--vision-bin`, and `--vlm-fixture`, the CLI runs:

```text
resized_rgb.u8 -> pixel_values -> ncnn vision -> text_embed injection -> decoder -> tokenizer
```

Fixture format:

```text
meta.txt                     # original/resized size, grid_t/grid_h/grid_w, patch_count
resized_rgb.u8               # uint8 RGB [resized_height, resized_width, 3]
expected_pixel_values.f32    # optional float32 [patch_count, 768]
```

The p512k deployment baseline uses `min_pixels=262144`,
`max_pixels=524288`, `patch_size=16`, and `merge_size=2`.

`--image-file-fixture` validates one step earlier, from original RGB bytes:

```text
meta.txt                     # original/resized size, grid_t/grid_h/grid_w, patch_count
original_rgb.u8              # uint8 RGB [original_height, original_width, 3]
expected_pixel_values.f32    # optional float32 [patch_count, 768]
```

`--image` decodes a PNG/JPEG file with vendored `stb_image`, resizes it with the
PIL-compatible bicubic path, and then runs the same preprocessing. Add
`--prompt-mode spotting` or `--prompt-mode document` to build the fixed
HunyuanOCR prompt tensors in C++. The CLI automatically selects a packaged
fixed-grid vision artifact under the model directory. When combined with
optional `--vlm-fixture`, the CLI can validate the current fixed-grid raw-image
development chain:

```text
PNG/JPEG image
-> RGB decode
-> PIL-compatible bicubic resize
-> pixel_values
-> ncnn fixed-grid vision
-> C++ built-in prompt + image token expansion + position_ids
-> text_embed injection
-> decoder
-> tokenizer
```

When `--prompt-mode` and `--vlm-fixture` are both passed, the fixture is used as
an oracle for expected `input_ids`, `position_ids`, generated tokens, and text.
The runtime path itself uses the C++ generated prompt tensors.

## Repository Layout

```text
include/hunyuan_ocr/   Public C++ headers
src/                   Runtime scaffold and CLI
third_party/           Vendored single-header dependencies such as stb_image
export/                Export script landing area
tools/                 Model packaging and verification tool landing area
examples/              Example usage notes
models/                Only example config is tracked; real models are ignored
```

## License

Original code in this repository is licensed under Apache-2.0. See `LICENSE`
and `NOTICE`. Vendored `stb_image.h` keeps its upstream MIT/Public Domain
license notice.

HunyuanOCR model files are governed by the Tencent Hunyuan Community License
Agreement and are not redistributed here.
