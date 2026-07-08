# Examples

This directory contains 28 public images used to exercise the runtime OCR path.
Image provenance is listed in `IMAGE_SOURCES.md`; reference output statistics
are listed in `EXPECTED_OUTPUTS.md`.

## Cases

| Case | Prompt mode | Image |
| --- | --- | --- |
| `hf_demo` | `spotting` | `hf_demo_tools-dark.png` |
| `chinese_doc` | `spotting` | `omnidoc_document_zh_page-205e4273-5b94-43e5-bfaf-dc882416b067.png` |
| `en_book` | `document` | `omnidoc_document_book_docstructbench_enbook_19221575_1173.jpg` |
| `formula` | `document` | `omnidoc_formula_harmonic_analysis_page_119.png` |
| `table` | `document` | `omnidoc_table_pyomo_page_188.png` |
| `hunyuan_subtitle_2` | `spotting` | `hunyuan_vis_subtitle2.png` |
| `hunyuan_table` | `document` | `hunyuan_vis_parsing_table.png` |
| `hunyuan_figure` | `document` | `hunyuan_vis_parsing_fig.png` |
| `hunyuan_subtitle` | `spotting` | `hunyuan_vis_subtitle1.png` |
| `hunyuan_spotting` | `spotting` | `hunyuan_spotting1_cropped.png` |
| `hunyuan_guwan1` | `spotting` | `hunyuan_guwan1.png` |
| `hunyuan_qikai1` | `spotting` | `hunyuan_qikai1.png` |
| `hunyuan_parsing_chart1` | `document` | `hunyuan_parsing_chart1.png` |
| `hunyuan_parsing_rgsj` | `document` | `hunyuan_parsing_rgsj.png` |
| `hunyuan_parsing_rgsjz_2` | `document` | `hunyuan_parsing_rgsjz_2.png` |
| `hunyuan_vis_parsing_chart1` | `document` | `hunyuan_vis_parsing_chart1.png` |
| `hunyuan_vis_parsing_chart2` | `document` | `hunyuan_vis_parsing_chart2.png` |
| `hunyuan_vis_parsing_table_2` | `document` | `hunyuan_vis_parsing_table_2.png` |
| `hunyuan_vis_subtitle3` | `spotting` | `hunyuan_vis_subtitle3.png` |
| `hunyuan_zimu2` | `spotting` | `hunyuan_zimu2.jpg` |
| `hunyuan_ie_parallel` | `document` | `hunyuan_ie_parallel.jpg` |
| `hunyuan_translation2` | `document` | `hunyuan_translation2.png` |
| `hunyuan_vis_translation` | `document` | `hunyuan_vis_translation.png` |
| `hunyuan_vis_ie_1` | `document` | `hunyuan_vis_ie_1.png` |
| `hunyuan_vis_parsing_chart3` | `document` | `hunyuan_vis_parsing_chart3.png` |
| `hunyuan_vis_art_16` | `spotting` | `hunyuan_vis_art_16.jpg` |
| `hunyuan_show_res_parsing_fig` | `document` | `hunyuan_show_res_parsing_fig.png` |
| `hunyuan_vis_parsing` | `document` | `hunyuan_vis_parsing.png` |

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

Run all bundled images:

```bash
python tools/run_examples.py \
  --model ./hunyuan_ocr_ncnn_model
```

`run_examples.py` writes per-case logs to `outputs/examples/`.

Dynamic vision packages support different image sizes within the exported
processor range with one `vision/vision.ncnn.param/bin` pair plus
`vision/pos_embed.bin`; fixed-grid fallback packages use `grid_<h>x<w>` names.

## Full Regression

After preparing exported fixtures, the full regression additionally compares
prompt ids, position ids, generated token ids, and decoded text:

```bash
python tools/run_regression.py --package --package-vision-backend dynamic
```

Expected summary:

```text
summary: 28/28 passed
```

## Custom Prompt Regression

`custom_prompt_cases.json` contains two small custom-prompt cases. Each case
uses the same image path as the bundled examples, but supplies a literal
`prompt` field instead of a built-in `prompt_mode`. See `tools/README.md` for
the PyTorch baseline, fixture preparation, and strict ncnn regression commands.
