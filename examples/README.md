# Examples

The CLI currently exposes development fixture paths. Model artifacts and
fixtures are intentionally kept outside git.

## Token Decode

```bash
./build/hunyuan_ocr_cli \
  --model /path/to/hunyuan_ocr_ncnn_model \
  --decode-ids-file generated_ids.txt
```

## Text Decoder Fixture

```bash
./build/hunyuan_ocr_cli \
  --model /path/to/hunyuan_ocr_ncnn_model \
  --text-fixture /tmp/text_fixture \
  --max-tokens 16
```

## VLM Fixture With External Vision Features

```bash
./build/hunyuan_ocr_cli \
  --model /path/to/hunyuan_ocr_ncnn_model \
  --vlm-fixture /tmp/vlm_fixture
```

## Vision Fixture Plus VLM Decode

This validates a fixed-grid vision ncnn artifact from flattened patch input,
then injects the generated features into the VLM decode fixture.

```bash
./build/hunyuan_ocr_cli \
  --model /path/to/hunyuan_ocr_ncnn_model \
  --vision-param /path/to/vision_fixed_grid.ncnn.param \
  --vision-bin /path/to/vision_fixed_grid.ncnn.bin \
  --vision-fixture /tmp/vision_fixture \
  --vlm-fixture /tmp/vlm_fixture
```

This is not yet a raw image OCR command. The current vision fixture expects
precomputed flattened patch tensors.

## Resized RGB Preprocess Plus VLM Decode

This validates C++ preprocessing from already resized RGB bytes to flattened
patch tensors, then runs the same fixed-grid vision and VLM decode path.

```bash
./build/hunyuan_ocr_cli \
  --model /path/to/hunyuan_ocr_ncnn_model \
  --image-preprocess-fixture /tmp/image_preprocess_fixture \
  --vision-param /path/to/vision_fixed_grid.ncnn.param \
  --vision-bin /path/to/vision_fixed_grid.ncnn.bin \
  --vlm-fixture /tmp/vlm_fixture
```

This still assumes the fixture image has already been resized with the same
PIL-compatible behavior as the HF processor.

## Raw PNG/JPEG Image Plus VLM Decode

This decodes a PNG/JPEG file in C++, applies the PIL-compatible resize and
HunyuanOCR patch flattening, then runs fixed-grid vision and the VLM decode
path. `--prompt-mode` builds the fixed prompt tensors in C++; the fixture is
only used as an oracle for expected output when provided.

```bash
./build/hunyuan_ocr_cli \
  --model /path/to/hunyuan_ocr_ncnn_model \
  --image /path/to/image.png \
  --prompt-mode spotting \
  --vlm-fixture /tmp/vlm_fixture
```

This is the current closest path to raw image OCR, but it is still a
development validation command. `--prompt-mode` currently supports the two
fixed baseline prompts, `spotting` and `document`; arbitrary user prompt
encoding is still pending.

The model directory must contain a fixed-grid vision package matching the
preprocessed image grid:

```text
vision/grid_<grid_h>x<grid_w>/vision.ncnn.param
vision/grid_<grid_h>x<grid_w>/vision.ncnn.bin
```
