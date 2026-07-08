# Export Workflow

This directory contains the scripts used to export HunyuanOCR HuggingFace
weights into ncnn modules consumed by `hunyuan_ocr_cli`.

The scripts are intentionally parameterized. Do not rely on author-local paths;
pass your HF model directory and pnnx executable explicitly.

## One Command Export

```bash
python export/export_all.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --workspace .
```

This writes:

```text
models/tokenizer/
models/export/text_embed/
models/export/text_decoder/
models/export/lm_head/
models/export/vision_dynamic_probe/
```

Then package the runtime directory:

```bash
python tools/package_model.py \
  --workspace . \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

Linux helper:

```bash
scripts/export_and_package_linux.sh \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn \
  --output ./hunyuan_ocr_ncnn_model \
  --copy
```

## Module Commands

Tokenizer:

```bash
python export/extract_tokenizer.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --out-dir models/tokenizer
```

Text embedding:

```bash
python export/export_text_embed.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --out-dir models/export/text_embed
```

LM head:

```bash
python export/export_lm_head.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --out-dir models/export/lm_head
```

Text decoder with KV cache:

```bash
python export/export_text_decoder_kv.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --out-dir models/export/text_decoder
```

Dynamic vision:

```bash
python export/export_vision_dynamic.py \
  --mode export \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --out-dir models/export/vision_dynamic_probe
```

`export_vision_dynamic.py` also keeps the original validation modes used during
development:

```bash
python export/export_vision_dynamic.py --mode torch ...
python export/export_vision_dynamic.py --mode ncnn ...
python export/export_vision_dynamic.py --mode token ...
```

The `ncnn` mode requires `--ncnn-python-dir /path/to/ncnn/python`.

## Export Workspace Layout

`tools/package_model.py` reads converted artifacts from a workspace with this
layout:

```text
<workspace>/
  models/
    tokenizer/
      vocab.txt
      merges.txt
      special_tokens.json
      eos_ids.json
    export/
      text_embed/
        text_embed.ncnn.param
        text_embed.ncnn.bin
      text_decoder/
        text_decoder_kv.ncnn.param
        text_decoder_kv.ncnn.bin
      lm_head/
        lm_head.ncnn.param
        lm_head.ncnn.bin
      vision_dynamic_probe/
        ncnn/
          vision_dynamic.ncnn.param
          vision_dynamic.ncnn.bin
          pos_embed.bin
      vision/
        fp32_p512k/
          summary.json
          ...
```

## Runtime Package Layout

The packager rewrites the export workspace into the runtime layout:

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
    grid_<h>x<w>/
      vision.ncnn.param
      vision.ncnn.bin
```

Dynamic vision uses `vision/vision.ncnn.param`, `vision/vision.ncnn.bin`, and
`vision/pos_embed.bin`. Fixed-grid fallback keeps directories such as
`vision/grid_38x52/`.

Create the package with:

```bash
python tools/package_model.py \
  --workspace <workspace> \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

Current validated exports use the fp32 path, `max_pixels=524288`, and
dynamic vision with fixed-grid fallback available. Use `--vision-backend fixed`
to reproduce the v0.1 fixed-grid-only package.
