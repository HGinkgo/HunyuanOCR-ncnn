# Tools

This directory contains model packaging, layout checks, and regression
verification helpers for the C++ runtime.

`package_model.py` builds the standard runtime model directory from exported
workspace artifacts. It creates symlinks by default and supports `--copy` for
portable bundles. The package verifies tied text embedding / LM Head weights
and stores their FP32 binary once. It always uses the canonical dynamic vision layout.
Use `--dflash` to include the optional DFlash draft
and auxiliary decoder; `--base-runtime-dir` can provide an existing 1.5 runtime
instead of the stock export layout.

`run_example.py` runs one bundled image from `examples/images/` through the
compiled CLI. Use `--list` to show available cases, and `--prompt` to pass a
custom image prompt instead of the case's built-in prompt mode.

`run_examples.py` runs all bundled example images and writes logs to
`outputs/examples/` by default.

`benchmark.py` runs one or more bundled image cases with `--benchmark`
enabled and reports stage timing such as preprocess, vision, prefill, decode,
and total runtime.

## Benchmark

`benchmark.py` runs bundled example images through `hunyuan_ocr_cli` with
`--benchmark` enabled. Each case/thread pair loads the model once, then runs
warmup and measured iterations in the same process.

Example:

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases hf_demo \
  --config cpu_fp32 \
  --threads 1,2,4,8,16 \
  --mmap-weights \
  --repeat 3 \
  --warmup 1 \
  --max-tokens 64 \
  --output-dir outputs/benchmark_hf_demo
```

Use `--vision-vulkan` and `--text-vulkan` to benchmark the optional backends;
their device indices default to zero. `--config` assigns a stable label to the
CSV and Markdown rows so results from separate model/backend configurations can
be compared without inferring the configuration from directory names. On an
otherwise idle NVIDIA device, `--nvidia-smi-device 0` also records the total
device-memory baseline, peak, and peak delta once per second.

Run all bundled cases:

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases all \
  --repeat 1 \
  --max-tokens 64
```

Reported fields:

- `mmap_weights`: `1` when read-only mapped weight loading is enabled.
- `mapped_weight_bytes`: total size of the retained model file mappings.
- `cold_start_total_ms`: process entry through the first completed inference.
- `warm_inference_total_ms`: image-to-text time with ncnn networks already loaded.
- `vision_load_ms`: one-time vision network and position embedding load.
- `text_load_ms`: one-time text subnetwork load.
- `preprocess_ms`: PNG/JPEG decode, resize, normalize, and patch flatten.
- `vision_infer_ms`: dynamic vision inference only.
- `prompt_ms`: tokenizer encode for custom prompts or built-in prompt assembly.
- `text_embed_ms`: prompt token embedding and image feature injection.
- `prefill_ms`: decoder cache-prefill pass.
- `decode_ms`: incremental decoder steps after the first generated token; with
  the fused Text Vulkan path, this includes incremental lm_head inference in
  the same submission.
- `lm_head_ms`: standalone lm_head inference; with fused Text Vulkan, only the
  first-token call remains standalone.
- `token_select_ms`: repetition penalty, argmax, and token selection.
- `tokenizer_decode_ms`: generated token ids to text.
- `decode_token_per_s`: incremental decoder-step throughput.
- `benchmark_process_wall_ms`: wall time measured by the Python parent process.
- `benchmark_process_peak_rss_bytes`: child-process peak RSS from Linux
  `VmHWM`.
- `benchmark_process_max_sampled_rss_anon_bytes`: maximum sampled anonymous
  RSS while the child process is alive.
- `benchmark_process_max_sampled_rss_file_bytes`: maximum sampled file-backed
  RSS while the child process is alive.
- `benchmark_process_memory_supported`: `1` when Linux process memory fields
  were available; unsupported platforms report zero-valued process memory
  fields and set this flag to `0`.
- `benchmark_device_memory_peak_delta_bytes`: peak total device memory minus
  the pre-launch baseline when `--nvidia-smi-device` is enabled.

The tool checks that generated token ids remain identical across all selected
thread counts for each case. `--output-dir` writes separate cold-start and warm
inference CSV files plus a compact Markdown summary.

Mapped pages count toward process RSS/PSS after they are touched. Use anonymous
and file-backed memory fields separately when comparing `--mmap-weights`; total
RSS alone does not represent private model memory.

The process memory fields do not include device-local Vulkan allocations. The
optional NVIDIA fields are device-level measurements, not per-process values;
NVIDIA process monitoring does not attribute this Vulkan workload reliably.

The script does not download models or prepare fixtures. It expects a packaged
runtime model directory produced by `tools/package_model.py`.

`run_regression.py` runs the bundled regression image cases against a
packaged model directory. It requires exported fixture directories from the
baseline/export workflow. Pass `--package` to rebuild the packaged model first;
the package always uses the canonical dynamic vision files. Manifest cases may
use either a built-in `prompt_mode` or a literal custom `prompt`. A
`max_tokens` value in the manifest is propagated through CPU baseline
generation, fixture metadata, and the ncnn CLI; mismatched stale fixtures are
rejected before inference.

`run_hf_baseline.py` creates PyTorch fp32 baseline outputs for a manifest. This
is mainly used to create strict regression fixtures for custom prompts:

```bash
python tools/run_hf_baseline.py \
  --model-dir /path/to/HunyuanOCR-hf \
  --manifest examples/custom_prompt_cases.json \
  --output-dir outputs/custom_prompt_baseline \
  --max-new-tokens 512

python tools/prepare_regression_fixtures.py \
  --baseline-dir outputs/custom_prompt_baseline \
  --manifest examples/custom_prompt_cases.json \
  --output-dir outputs/custom_prompt_fixtures \
  --force

python tools/run_regression.py \
  --model ./hunyuan_ocr_ncnn_model \
  --manifest examples/custom_prompt_cases.json \
  --fixture-root outputs/custom_prompt_fixtures
```
