# Tools

This directory contains model packaging, layout checks, and regression
verification helpers for the C++ runtime.

`package_model.py` builds the standard runtime model directory from exported
workspace artifacts. It creates symlinks by default and supports `--copy` for
portable bundles. Use `--vision-backend fixed`, `dynamic`, or `both` to choose
the packaged vision layout.

`run_example.py` runs one bundled image from `examples/images/` through the
compiled CLI. Use `--list` to show available cases, and `--prompt` to pass a
custom image prompt instead of the case's built-in prompt mode.

`run_examples.py` runs all bundled example images and writes logs to
`outputs/examples/` by default.

`benchmark.py` runs one or more bundled image cases with `--benchmark`
enabled and reports stage timing such as preprocess, vision, prefill, decode,
and total runtime.

`run_regression.py` runs the bundled regression image cases against a
packaged model directory. It requires exported fixture directories from the
baseline/export workflow. Pass `--package` to rebuild the packaged model first;
`--package-vision-backend dynamic` runs the same regression through the dynamic
vision package. Manifest cases may use either a built-in `prompt_mode` or a
literal custom `prompt`.

`run_hf_baseline.py` creates PyTorch fp32 baseline outputs for a manifest. This
is mainly used to create strict regression fixtures for custom prompts:

```bash
python tools/run_hf_baseline.py \
  --model-dir /path/to/HunyuanOCR-hf \
  --manifest examples/custom_prompt_cases.json \
  --output-dir outputs/custom_prompt_baseline \
  --max-new-tokens 128

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
