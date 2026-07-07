# Export Notes

This directory documents the pnnx export outputs expected by the runtime. The
export workspace keeps intermediate conversion files, while the runtime package
contains only the files loaded by `hunyuan_ocr_cli`.

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
