# Benchmark

`tools/benchmark.py` runs bundled example images through `hunyuan_ocr_cli`
with `--benchmark` enabled and reports average stage timing.

Example:

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases hf_demo \
  --repeat 3 \
  --warmup 1 \
  --max-tokens 64 \
  --csv outputs/benchmark_hf_demo.csv
```

Run all bundled cases:

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases all \
  --repeat 1 \
  --max-tokens 64
```

Reported fields:

- `preprocess_ms`: PNG/JPEG decode, resize, normalize, and patch flatten.
- `vision_ms`: vision network load and inference for the image.
- `prompt_ms`: tokenizer encode for custom prompts or built-in prompt assembly.
- `text_load_ms`: text subnetwork load time.
- `text_embed_ms`: prompt token embedding and image feature injection.
- `prefill_ms`: decoder cache-prefill pass.
- `decode_ms`: incremental decode loop after the first generated token.
- `total_ms`: end-to-end image-to-text runtime in the CLI process.
- `decode_token_per_s`: incremental decode throughput.

The script does not download models or prepare fixtures. It expects a packaged
runtime model directory produced by `tools/package_model.py`.
