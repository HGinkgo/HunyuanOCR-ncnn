# Examples

This directory contains a small set of public images used to exercise the
runtime OCR path. Image provenance is listed in `IMAGE_SOURCES.md`; short
reference output excerpts are listed in `EXPECTED_OUTPUTS.md`.

## Cases

| Case | Prompt mode | Image |
| --- | --- | --- |
| `hf_demo` | `spotting` | `hf_demo_tools-dark.png` |
| `chinese_doc` | `spotting` | `omnidoc_document_zh_page-205e4273-5b94-43e5-bfaf-dc882416b067.png` |
| `en_book` | `document` | `omnidoc_document_book_docstructbench_enbook_19221575_1173.jpg` |
| `formula` | `document` | `omnidoc_formula_harmonic_analysis_page_119.png` |
| `table` | `document` | `omnidoc_table_pyomo_page_188.png` |

## Run

These commands assume `hunyuan_ocr_cli` has been built under `build/` and the
runtime model has been packaged as `./hunyuan_ocr_ncnn_model`.

List the bundled examples:

```bash
python tools/run_example.py --list
```

Run one example:

```bash
python tools/run_example.py \
  --model ./hunyuan_ocr_ncnn_model \
  --case hf_demo
```

Run all bundled examples:

```bash
python tools/run_examples.py \
  --model ./hunyuan_ocr_ncnn_model
```

`run_examples.py` writes per-case logs to `outputs/examples/`.

The image must preprocess to a supported grid. Runtime vision directories use
the `grid_<h>x<w>` naming convention, for example `image_grid_thw=[1,38,52]`
uses `vision/grid_38x52/`.

## Full Regression

After preparing exported fixtures, the five-sample regression additionally
compares prompt ids, position ids, generated token ids, and decoded text:

```bash
python tools/run_5sample_regression.py --package
```

Expected summary:

```text
summary: 5/5 passed
```
