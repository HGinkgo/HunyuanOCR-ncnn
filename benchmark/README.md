# Benchmark

`tools/benchmark.py` runs bundled example images through `hunyuan_ocr_cli`
with `--benchmark` enabled. Each case/thread pair loads the model once, then
runs warmup and measured iterations in the same process.

Example:

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases hf_demo \
  --threads 1,2,4,8,16 \
  --repeat 3 \
  --warmup 1 \
  --max-tokens 64 \
  --output-dir outputs/benchmark_hf_demo
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

- `cold_start_total_ms`: process entry through the first completed inference.
- `warm_inference_total_ms`: image-to-text time with ncnn networks already loaded.
- `vision_load_ms`: one-time vision network and position embedding load.
- `text_load_ms`: one-time text subnetwork load.
- `preprocess_ms`: PNG/JPEG decode, resize, normalize, and patch flatten.
- `vision_infer_ms`: dynamic/fixed vision inference only.
- `prompt_ms`: tokenizer encode for custom prompts or built-in prompt assembly.
- `text_embed_ms`: prompt token embedding and image feature injection.
- `prefill_ms`: decoder cache-prefill pass.
- `decode_ms`: incremental decoder steps after the first generated token.
- `lm_head_ms`: first-token and incremental lm_head inference.
- `token_select_ms`: repetition penalty, argmax, and token selection.
- `tokenizer_decode_ms`: generated token ids to text.
- `decode_token_per_s`: incremental decoder-step throughput.

The tool checks that generated token ids remain identical across all selected
thread counts for each case. `--output-dir` writes separate cold-start and warm
inference CSV files plus a compact Markdown summary.

The script does not download models or prepare fixtures. It expects a packaged
runtime model directory produced by `tools/package_model.py`.
