# Examples

These examples assume:

- `hunyuan_ocr_cli` has been built under `build/`.
- Runtime artifacts have been packaged with `tools/package_model.py`.
- Model files, fixtures, and outputs remain outside git.

## Package The Runtime Model

```bash
python tools/package_model.py \
  --workspace /root/hpf/workspace/ncnn_hunyuanocr \
  --output /tmp/hunyuanocr_ncnn_model_packaged \
  --force
```

Use `--copy` if you need a portable model directory instead of symlinks.

## Run A PNG/JPEG Image

Spot text and coordinates:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --image /root/hpf/workspace/ncnn_hunyuanocr/datasets/test_images/hf_demo_tools-dark.png \
  --prompt-mode spotting
```

Parse a document-style image into markdown-like text:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --image /path/to/document.png \
  --prompt-mode document
```

The image must match a packaged fixed-grid vision directory:

```text
vision/grid_<grid_h>x<grid_w>/vision.ncnn.param
vision/grid_<grid_h>x<grid_w>/vision.ncnn.bin
```

## Run The Five-Sample Regression

```bash
python tools/run_5sample_regression.py --package
```

This rebuilds `/tmp/hunyuanocr_ncnn_model_packaged`, runs the five local golden
images, compares prompt ids, position ids, generated token ids, and decoded
text, then writes logs to `/tmp/hunyuanocr_5sample_regression/`.

Expected summary:

```text
summary: 5/5 passed
```

## Development Validation Commands

These commands are for debugging individual runtime stages. They require
fixtures produced from the Python baseline/export workflow.

Decode generated token ids:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --decode-ids-file generated_ids.txt
```

Run decoder prefill/KV decode from raw tensors:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --text-fixture /tmp/text_fixture \
  --max-tokens 16
```

Run text embedding, vision feature injection, decoder, and tokenizer decode:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --vlm-fixture /tmp/vlm_fixture
```

Validate a fixed-grid vision artifact from flattened patch tensors, then feed
the resulting vision features into the VLM fixture:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --vision-param /path/to/vision_fixed_grid.ncnn.param \
  --vision-bin /path/to/vision_fixed_grid.ncnn.bin \
  --vision-fixture /tmp/vision_fixture \
  --vlm-fixture /tmp/vlm_fixture
```

Validate resized RGB preprocessing before vision and decode:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --image-preprocess-fixture /tmp/image_preprocess_fixture \
  --vision-param /path/to/vision_fixed_grid.ncnn.param \
  --vision-bin /path/to/vision_fixed_grid.ncnn.bin \
  --vlm-fixture /tmp/vlm_fixture
```

Validate original RGB resize, preprocessing, vision, and decode:

```bash
./build/hunyuan_ocr_cli \
  --model /tmp/hunyuanocr_ncnn_model_packaged \
  --image-file-fixture /tmp/image_file_fixture \
  --vision-param /path/to/vision_fixed_grid.ncnn.param \
  --vision-bin /path/to/vision_fixed_grid.ncnn.bin \
  --vlm-fixture /tmp/vlm_fixture
```

When `--vlm-fixture` is passed with image or vision commands, the fixture is
used as an oracle for expected ids/tokens/text. The runtime path still uses the
C++ generated tensors for the stages under test.
