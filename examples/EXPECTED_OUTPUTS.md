# Example Output Summary

These statistics come from the HunyuanOCR 1.5 Transformers 5.13.0 CPU fp32
eager reference at checkpoint revision
`9e01f897bf8956f77a80c350dc0491d6bbbd43e6`. The processor range is fixed to
`min_pixels=262144` and `max_pixels=524288`, and generation uses greedy decode
with repetition penalty `1.08`. The table records generation within the
validated 512-token safety window. `New tokens` is the actual generated count,
so cases that reach EOS early remain shorter than 512. These are model reference
outputs, not manually corrected OCR ground truth.

| Case | Prompt mode | Grid | New tokens | Output chars |
| --- | --- | --- | ---: | ---: |
| `hf_demo` | `spotting` | `38x52` | 439 | 1186 |
| `chinese_doc` | `spotting` | `54x36` | 512 | 748 |
| `en_book` | `document` | `58x34` | 512 | 1844 |
| `formula` | `document` | `58x34` | 512 | 1447 |
| `table` | `document` | `54x36` | 512 | 877 |
| `hunyuan_subtitle_2` | `spotting` | `20x94` | 128 | 196 |
| `hunyuan_table` | `document` | `52x38` | 512 | 1027 |
| `hunyuan_figure` | `document` | `32x60` | 512 | 886 |
| `hunyuan_subtitle` | `spotting` | `24x84` | 127 | 214 |
| `hunyuan_spotting` | `spotting` | `34x56` | 146 | 267 |
| `hunyuan_guwan1` | `spotting` | `28x70` | 101 | 145 |
| `hunyuan_qikai1` | `spotting` | `36x56` | 59 | 88 |
| `hunyuan_parsing_chart1` | `document` | `32x60` | 40 | 92 |
| `hunyuan_parsing_rgsj` | `document` | `32x62` | 47 | 84 |
| `hunyuan_parsing_rgsjz_2` | `document` | `16x92` | 23 | 42 |
| `hunyuan_vis_parsing_chart1` | `document` | `36x54` | 329 | 729 |
| `hunyuan_vis_parsing_chart2` | `document` | `34x56` | 427 | 893 |
| `hunyuan_vis_parsing_table_2` | `document` | `46x42` | 512 | 1227 |
| `hunyuan_vis_subtitle3` | `spotting` | `44x44` | 141 | 220 |
| `hunyuan_zimu2` | `spotting` | `32x60` | 44 | 74 |
| `hunyuan_ie_parallel` | `document` | `58x22` | 141 | 267 |
| `hunyuan_translation2` | `document` | `40x48` | 512 | 932 |
| `hunyuan_vis_translation` | `document` | `34x56` | 504 | 1147 |
| `hunyuan_vis_ie_1` | `document` | `46x42` | 512 | 1019 |
| `hunyuan_vis_parsing_chart3` | `document` | `32x60` | 428 | 852 |
| `hunyuan_vis_art_16` | `spotting` | `40x40` | 29 | 42 |
| `hunyuan_show_res_parsing_fig` | `document` | `44x46` | 512 | 1020 |
| `hunyuan_vis_parsing` | `document` | `36x54` | 512 | 1546 |

The ncnn CPU fp32 run preserves OCR text and structure across this safety
window. `hunyuan_figure` is the only non-token-exact numerical boundary: after
bbox removal its OCR text and structure are identical, while 10 of 18 boxes
differ with a maximum coordinate delta of 3 and a mean absolute coordinate
delta of 0.361. The difference is disclosed rather than hidden by coordinate
compensation.
