# Example Output Summary

These are PyTorch fp32 reference output statistics for the current
`max_pixels=524288` validation set. They are model reference outputs, not
manually corrected OCR ground truth. The C++/ncnn regression compares the full
generated token sequence and decoded text through fixture files.

| Case | Prompt mode | Grid | New tokens | Output chars |
| --- | --- | --- | ---: | ---: |
| `hf_demo` | `spotting` | `38x52` | 411 | 1150 |
| `chinese_doc` | `spotting` | `54x36` | 593 | 859 |
| `en_book` | `document` | `58x34` | 1024 | 3437 |
| `formula` | `document` | `58x34` | 979 | 2961 |
| `table` | `document` | `54x36` | 325 | 1770 |
| `hunyuan_subtitle_2` | `spotting` | `20x94` | 172 | 257 |
| `hunyuan_table` | `document` | `52x38` | 1024 | 1907 |
| `hunyuan_figure` | `document` | `32x60` | 24 | 67 |
| `hunyuan_subtitle` | `spotting` | `24x84` | 130 | 219 |
| `hunyuan_spotting` | `spotting` | `34x56` | 188 | 324 |
| `hunyuan_guwan1` | `spotting` | `28x70` | 101 | 145 |
| `hunyuan_qikai1` | `spotting` | `36x56` | 60 | 89 |
| `hunyuan_parsing_chart1` | `document` | `32x60` | 40 | 92 |
| `hunyuan_parsing_rgsj` | `document` | `32x62` | 42 | 70 |
| `hunyuan_parsing_rgsjz_2` | `document` | `16x92` | 34 | 64 |
| `hunyuan_vis_parsing_chart1` | `document` | `36x54` | 182 | 386 |
| `hunyuan_vis_parsing_chart2` | `document` | `34x56` | 451 | 946 |
| `hunyuan_vis_parsing_table_2` | `document` | `46x42` | 1024 | 2370 |
| `hunyuan_vis_subtitle3` | `spotting` | `44x44` | 126 | 195 |
| `hunyuan_zimu2` | `spotting` | `32x60` | 46 | 77 |
| `hunyuan_ie_parallel` | `document` | `58x22` | 158 | 310 |
| `hunyuan_translation2` | `document` | `40x48` | 844 | 1537 |
| `hunyuan_vis_translation` | `document` | `34x56` | 578 | 1880 |
| `hunyuan_vis_ie_1` | `document` | `46x42` | 541 | 1212 |
| `hunyuan_vis_parsing_chart3` | `document` | `32x60` | 627 | 1263 |
| `hunyuan_vis_art_16` | `spotting` | `40x40` | 29 | 42 |
| `hunyuan_show_res_parsing_fig` | `document` | `44x46` | 1024 | 2114 |
| `hunyuan_vis_parsing` | `document` | `36x54` | 1024 | 2526 |
