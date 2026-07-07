# Example Image Sources

The example images in `examples/images/` are the small public image set used for
the current fp32 `max_pixels=524288` validation.

| File | Prompt mode | Source |
| --- | --- | --- |
| `hf_demo_tools-dark.png` | `spotting` | HunyuanOCR README demo image URL used during baseline setup: `chat-ui/tools-dark.png`. Source project: https://github.com/Tencent-Hunyuan/HunyuanOCR |
| `omnidoc_document_zh_page-205e4273-5b94-43e5-bfaf-dc882416b067.png` | `spotting` | OmniDocBench sample. Source project: https://github.com/opendatalab/OmniDocBench |
| `omnidoc_document_book_docstructbench_enbook_19221575_1173.jpg` | `document` | OmniDocBench sample. Source project: https://github.com/opendatalab/OmniDocBench |
| `omnidoc_formula_harmonic_analysis_page_119.png` | `document` | OmniDocBench sample. Source project: https://github.com/opendatalab/OmniDocBench |
| `omnidoc_table_pyomo_page_188.png` | `document` | OmniDocBench sample. Source project: https://github.com/opendatalab/OmniDocBench |
| `hunyuan_vis_subtitle2.png` | `spotting` | HunyuanOCR upstream asset `assets/vis_subtitle2.png`. Source project: https://github.com/Tencent-Hunyuan/HunyuanOCR |
| `hunyuan_vis_parsing_table.png` | `document` | HunyuanOCR upstream asset `assets/vis_parsing_table.png`. Source project: https://github.com/Tencent-Hunyuan/HunyuanOCR |
| `hunyuan_vis_parsing_fig.png` | `document` | HunyuanOCR upstream asset `assets/vis_parsing_fig.png`. Source project: https://github.com/Tencent-Hunyuan/HunyuanOCR |
| `hunyuan_vis_subtitle1.png` | `spotting` | HunyuanOCR upstream asset `assets/vis_subtitle1.png`. Source project: https://github.com/Tencent-Hunyuan/HunyuanOCR |
| `hunyuan_spotting1_cropped.png` | `spotting` | HunyuanOCR upstream asset `assets/spotting1_cropped.png`. Source project: https://github.com/Tencent-Hunyuan/HunyuanOCR |

OmniDocBench is used here only as a small public document-image regression
source. Follow the original dataset terms for any use beyond local validation
or reproduction.
