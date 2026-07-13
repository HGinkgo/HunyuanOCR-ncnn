# Model Package Layout

Runtime model files are generated outside the source tree and passed to
`hunyuan_ocr_cli` with `--model`.

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
  dflash/                       # optional
    dflash.ncnn.param
    dflash.ncnn.bin
  vision/
    vision.ncnn.param
    vision.ncnn.bin
    pos_embed.bin
    grid_<grid_h>x<grid_w>/
      vision.ncnn.param
      vision.ncnn.bin
```

## Required Files

The current runtime requires tokenizer files, `text_embed`, `text_decoder`,
`lm_head`, and one supported vision backend.

`model.json` records the expected relative paths. See
`models/model.json.example` for the current schema.

The `main` branch `0.4.0` preview template targets HunyuanOCR 1.5 checkpoint revision
`9e01f897bf8956f77a80c350dc0491d6bbbd43e6` with repetition penalty `1.08`
and EOS token `120020`. Branch `feat/hunyuanocr-1.0` preserves the HunyuanOCR
1.0 development line, while tag `v0.2.0` remains its frozen release.

The optional DFlash package adds `dflash/dflash.ncnn.param/bin` and replaces the
base decoder param with the auxiliary export that exposes `out1` through `out4`.
The decoder weights are unchanged. Packages without these optional files continue
to use the default AR path.

## Vision Backends

`tools/package_model.py --vision-backend dynamic` creates:

```text
vision/vision.ncnn.param
vision/vision.ncnn.bin
vision/pos_embed.bin
```

This is the default path. It supports image grids inside the exported processor
range and interpolates the base position embedding at runtime.

`tools/package_model.py --vision-backend fixed` creates per-grid directories:

```text
vision/grid_38x52/vision.ncnn.param
vision/grid_38x52/vision.ncnn.bin
```

Fixed-grid packages only run images whose `image_grid_thw=[1,h,w]` has a
matching `vision/grid_<h>x<w>/` directory.

`tools/package_model.py --vision-backend both` includes dynamic vision and the
fixed-grid fallback files in the same package.

## Packaging

```bash
python tools/package_model.py \
  --workspace <workspace> \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

Here `<workspace>` is the directory containing `models/tokenizer/` and
`models/export/`. The generated model package can be moved independently when
`--copy` is used.

Pass `--dflash` to include the draft network and auxiliary decoder. Existing base
runtime files may be supplied with `--base-runtime-dir`; custom draft and decoder
locations use `--dflash-dir` and `--dflash-decoder-dir`.
