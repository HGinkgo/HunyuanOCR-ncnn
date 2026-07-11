# Example Output Summary

These statistics come from the HunyuanOCR 1.5 Transformers 5.13.0 CPU fp32
eager reference at checkpoint revision
`9e01f897bf8956f77a80c350dc0491d6bbbd43e6`. The processor range is fixed to
`min_pixels=262144` and `max_pixels=524288`, and generation uses greedy decode
with repetition penalty `1.08`. The table records the validated 128-token
window; these are model reference outputs, not manually corrected OCR ground
truth.

| Case | Prompt mode | Grid | New tokens | Output chars |
| --- | --- | --- | ---: | ---: |
| `hf_demo` | `spotting` | `38x52` | 128 | 268 |
| `chinese_doc` | `spotting` | `54x36` | 128 | 186 |
| `en_book` | `document` | `58x34` | 128 | 407 |
| `formula` | `document` | `58x34` | 128 | 349 |
| `table` | `document` | `54x36` | 128 | 223 |
| `hunyuan_subtitle_2` | `spotting` | `20x94` | 128 | 196 |
| `hunyuan_table` | `document` | `52x38` | 128 | 291 |
| `hunyuan_figure` | `document` | `32x60` | 128 | 237 |
| `hunyuan_subtitle` | `spotting` | `24x84` | 128 | 217 |
| `hunyuan_spotting` | `spotting` | `34x56` | 128 | 240 |
| `hunyuan_guwan1` | `spotting` | `28x70` | 128 | 176 |
| `hunyuan_qikai1` | `spotting` | `36x56` | 128 | 186 |
| `hunyuan_parsing_chart1` | `document` | `32x60` | 128 | 228 |
| `hunyuan_parsing_rgsj` | `document` | `32x62` | 128 | 198 |
| `hunyuan_parsing_rgsjz_2` | `document` | `16x92` | 128 | 213 |
| `hunyuan_vis_parsing_chart1` | `document` | `36x54` | 128 | 307 |
| `hunyuan_vis_parsing_chart2` | `document` | `34x56` | 128 | 286 |
| `hunyuan_vis_parsing_table_2` | `document` | `46x42` | 128 | 305 |
| `hunyuan_vis_subtitle3` | `spotting` | `44x44` | 128 | 201 |
| `hunyuan_zimu2` | `spotting` | `32x60` | 128 | 187 |
| `hunyuan_ie_parallel` | `document` | `58x22` | 128 | 247 |
| `hunyuan_translation2` | `document` | `40x48` | 128 | 265 |
| `hunyuan_vis_translation` | `document` | `34x56` | 128 | 190 |
| `hunyuan_vis_ie_1` | `document` | `46x42` | 128 | 258 |
| `hunyuan_vis_parsing_chart3` | `document` | `32x60` | 128 | 275 |
| `hunyuan_vis_art_16` | `spotting` | `40x40` | 128 | 87 |
| `hunyuan_show_res_parsing_fig` | `document` | `44x46` | 128 | 243 |
| `hunyuan_vis_parsing` | `document` | `36x54` | 128 | 368 |
