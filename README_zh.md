<div align="center">
  <h1>HunyuanOCR-ncnn</h1>
  <p>
    基于 <a href="https://huggingface.co/tencent/HunyuanOCR"><strong>HunyuanOCR 1.5</strong></a>
    和 <a href="https://github.com/Tencent/ncnn"><strong>ncnn</strong></a> 的纯 C++17 OCR 推理运行时
  </p>
  <p>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/linux-ci.yml/badge.svg" alt="Linux CI"></a>
    <a href="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml"><img src="https://github.com/HGinkgo/HunyuanOCR-ncnn/actions/workflows/windows-compile.yml/badge.svg" alt="Windows CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
    <img src="https://img.shields.io/badge/C%2B%2B-17-f34b7d.svg" alt="C++17">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg" alt="Linux and Windows">
    <img src="https://img.shields.io/badge/backend-CPU%20fp32%20%7C%20Vulkan%20vision%20fp32-4c1.svg" alt="CPU fp32 和 Vulkan vision fp32">
  </p>
  <p>
    Tencent/ncnn 开源活动作品 <a href="https://github.com/Tencent/ncnn/discussions/6808">#6808</a>
    &nbsp;|&nbsp; <a href="README.md">English</a>
  </p>
</div>

---

本仓库使用 pnnx 将 Hugging Face 版 HunyuanOCR 导出为 ncnn 子模块，并在 C++ 中跑通完整 OCR 推理链路。

> **HunyuanOCR 1.5 预览版本（`0.4.0`）：** 当前开发分支固定 checkpoint
> revision `9e01f897bf8956f77a80c350dc0491d6bbbd43e6`，严格参考使用
> Transformers 5.13.0、CPU fp32 和 eager attention。Linux、Windows 验证及
> 28-case 测试均已通过；
> `v0.2.0` 继续作为冻结的 HunyuanOCR 1.0 版本。

| 开发线 | 模型 | 状态 |
| --- | --- | --- |
| `main` | HunyuanOCR 1.5 | `0.4.0` development/preview |
| `feat/hunyuanocr-1.0` | HunyuanOCR 1.0 | 保留的兼容分支 |
| `v0.2.0` | HunyuanOCR 1.0 | 冻结版本 |

## 比赛要求覆盖

| 任务要求 | 仓库内证据 |
| --- | --- |
| 使用 pnnx 转换 HunyuanOCR | `export/` 提供可复现的子模块导出流程，`tools/package_model.py` 负责运行时模型打包 |
| 参考 ncnn_llm，以少量依赖实现 C++ LLM 解码 | runtime 和 KV-cache decoder 以 [ncnn_llm](https://github.com/nihui/ncnn_llm) 为架构参考；C++17 可执行程序使用 ncnn 以及仓库内的 `stb_image`、`picojson` |
| 最终文本与 PyTorch 原版一致 | 固定的 Transformers 5.13.0 CPU fp32 参考与 ncnn runtime 在已验证的 128-token 窗口内通过全部 28 个 token/text case |
| CMake 至少覆盖两个平台 | CI 覆盖 Linux、Windows 构建和轻量测试；带模型验证已在两个平台独立完成 |
| 发表技术总结并提供仓库地址 | [Tencent/ncnn Discussion #6808](https://github.com/Tencent/ncnn/discussions/6808) 已链接本仓库 |

HunyuanOCR 1.5 适配及其验证由本仓库维护；上面的 ncnn_llm 链接用于明确记录
任务要求中的参考项目，并不表示本仓库是 ncnn_llm 的上游分支。

## 扩展能力

- 支持已导出范围内的不同图片尺寸，并保留 fixed-grid 回退包。
- append-only、带容量的 KV cache 避免稳态 past-cache 拷贝，并以独立生命周期测试
  覆盖扩容、逻辑 view 和重复请求。
- 内置 `spotting` / `document` 两种模式，也支持自定义 UTF-8 `--prompt` 文本。
- 可复用 C++ runtime 在连续请求间保持模型网络常驻；严格 JSONL 批处理保持输入顺序，
  并为每条失败记录输出独立错误结果。
- Windows CLI 全链路支持 UTF-8 prompt、模型路径、图片路径和 fixture 路径。
- 可选 DFlash speculative decoding，默认推理路径仍为 AR。
- 可选 fp32 Vulkan vision 后端；使用项目维护的 ncnn 补丁集时，28-case
  token/text 测试通过，且 GELU 不回退到 CPU。
- 100 轮 RSS 回归和 ASAN/UBSAN 门禁审计长期存活的 vision、text、DFlash
  runtime 是否在请求结束后释放 ncnn 临时缓冲区。

## 常用入口

| 需求 | 入口 |
| --- | --- |
| 构建并跑一张图 | `scripts/quickstart_existing_model.sh` |
| 打包转换产物 | `tools/package_model.py` |
| 从 HF 权重导出 | `export/README.md` |
| 运行示例图 | `tools/run_example.py`, `tools/run_examples.py` |
| 完整测试 | `tools/run_regression.py` |
| 性能测试 | `tools/benchmark.py`, `tools/README.md#benchmark` |
| 模型目录协议 | `models/README.md`, `models/model.json.example` |

当前已验证配置使用 `max_pixels=524288`。这是因为开发和验证硬件资源为
RTX 3090（24 GB），需要控制模型转换与测试过程的显存占用；它表示当前项目的
已验证范围，并不是 HunyuanOCR 或 ncnn 的理论上限。就运行方式而言，dynamic
vision 包可覆盖已导出范围内的不同图片尺寸，fixed-grid 包主要作为兼容回退。
模型目录和字段说明见 `models/README.md`。

当前版本暂不覆盖原版高分辨率路径。

对于对 JPEG 解码舍入敏感的 case，严格测试使用规范化无损 PNG。运行时仍
支持 JPEG，但 Pillow/libjpeg-turbo 与 `stb_image` 对有损 JPEG 的解码像素可能
存在微小差异，因此不承诺任意 JPEG 都能跨解码器保持 token 完全一致。

## 实验性 DFlash

`--dflash` 可以为 greedy generation 显式启用 DFlash speculative decoder。模型包
需要包含 `dflash/dflash.ncnn.param/bin`，并使用导出 `out1` 至 `out4` 的辅助
text decoder。AR 仍是默认路径，运行时不会自动切换解码方式。

Linux 和 Windows 验证均保持与 AR 的 token/text 输出一致。性能收益取决于输入：
当前三个 CPU case 的 warm speedup 随 draft acceptance 从 `2.71%` 增长到
`17.04%`，范围为 `0.64x` 至 `1.20x`。低 acceptance 输入可能更慢，因此
DFlash 仍是显式启用的 development/preview 能力，不作为通用加速承诺。

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode document \
  --dflash
```

## 快速开始

如果已经准备好打包后的 `hunyuan_ocr_ncnn_model/`，可以用最短路径构建并跑一张示例图：

```bash
scripts/quickstart_existing_model.sh \
  --model ./hunyuan_ocr_ncnn_model \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn
```

如果已经构建完成，只想跑一张图做 smoke test：

```bash
scripts/smoke_test.sh --model ./hunyuan_ocr_ncnn_model
```

仓库不包含模型权重。已有运行时模型包的用户可以直接使用上述入口；如果需要从
Hugging Face 模型重新完成转换，请按下面的“导出 → 打包 → 构建 → 运行”流程操作。

## 从 HF 模型导出

导出脚本位于 `export/`，使用 pnnx 将 Hugging Face 模型转换为 tokenizer 和 ncnn
子模块产物：

```bash
python export/export_all.py \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --workspace .
```

模块级命令见 `export/README.md`。

## 打包 ncnn 运行时模型

`tools/package_model.py` 将导出产物整理为 CLI 可以直接加载的运行时模型包：

```text
hunyuan_ocr_ncnn_model/
  model.json
  tokenizer/
  text_embed/
  text_decoder/
  lm_head/
  vision/
```

```bash
python tools/package_model.py \
  --workspace <workspace> \
  --output ./hunyuan_ocr_ncnn_model \
  --vision-backend dynamic \
  --copy \
  --force
```

增加 `--dflash` 可从默认导出目录打包 draft network 和辅助 decoder；也可以用
`--dflash-dir`、`--dflash-decoder-dir` 和 `--base-runtime-dir` 显式指定来源。

这里 `<workspace>` 指包含 `models/tokenizer/` 和 `models/export/` 的工作目录。
如果需要 v0.1 fixed-grid 包，使用 `--vision-backend fixed`；如果需要同时包含
dynamic vision 和 fixed-grid fallback，使用 `--vision-backend both`。
`models/README.md` 说明了 `model.json` 字段和 dynamic/fixed vision backend 的选择规则。

Linux 下也可以使用封装脚本一次完成构建、导出、打包和示例运行：

```bash
scripts/export_and_package_linux.sh \
  --hf-dir /path/to/HunyuanOCR-hf \
  --pnnx /path/to/pnnx \
  --ncnn-dir /path/to/ncnn/lib/cmake/ncnn \
  --output ./hunyuan_ocr_ncnn_model \
  --copy
```

## 构建

依赖：

- CMake 3.18 或更新版本
- 支持 C++17 的编译器
- ncnn `20260106` 或更新版本；验证固定 revision 为
  `244f30c8b995d5b2cf57b59950596490c68813d6`

默认 CPU 构建可以直接使用未修改的固定 ncnn checkout。可选的 fp32 Vulkan
vision 后端使用本项目维护的 [`patches/ncnn`](patches/ncnn) 补丁集：

```bash
python scripts/apply_ncnn_patches.py --ncnn-dir /path/to/ncnn
```

其中 Vulkan MatMul 实现源自 Cat-myq 提交的
[Tencent/ncnn PR #6579](https://github.com/Tencent/ncnn/pull/6579)，对应 commit
`88e0927f6e6b640fea19bd5721ff5409fcca99ef`；本文不将该 PR 描述为已经合入
上游。第二个补丁为 HunyuanOCR vision encoder 增加所需的 exact fp32 GELU
路径。补丁包和 `NOTICE` 保留了来源说明以及 ncnn BSD 3-Clause 许可证。

如果已经安装 ncnn CMake package：

```bash
cmake -S . -B build -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build build -j
```

如果使用本地 ncnn build 目录：

```bash
cmake -S . -B build \
  -DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF \
  -DNCNN_INCLUDE_DIR=/path/to/ncnn/src \
  -DNCNN_BUILD_INCLUDE_DIR=/path/to/ncnn/build/src \
  -DNCNN_LIBRARY=/path/to/ncnn/build/src/libncnn.a
cmake --build build -j
```

基础检查：

```bash
./build/hunyuan_ocr_cli --help
./build/hunyuan_ocr_cli --version
```

构建补丁版 ncnn SDK 后，可以让 vision encoder 使用 Vulkan：

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --image ./examples/images/hf_demo_tools-dark.png \
  --prompt-mode spotting \
  --vision-vulkan \
  --vision-vulkan-device 0
```

当前仅 vision encoder 使用 Vulkan；text generation 仍使用原有 CPU fp32
路径。

Windows 构建和带模型验证均已通过。CLI 从宽字符命令行读取参数，支持中文等
Unicode prompt 和模型、图片、fixture 路径；控制台输入输出统一使用 UTF-8。

## 可复用 C++ Runtime

`hunyuan_ocr` 静态库目标持有已加载的模型，可以连续处理多次请求而不重新加载
ncnn 网络。源码方式接入如下：

```cmake
add_subdirectory(path/to/HunyuanOCR-ncnn)
target_link_libraries(my_ocr_app PRIVATE hunyuan_ocr)
```

```cpp
#include "hunyuan_ocr/hunyuan_ocr.h"

hunyuan_ocr::RuntimeOptions options;
options.num_threads = 8;

hunyuan_ocr::RuntimeError error;
hunyuan_ocr::HunyuanOCR runtime;
if (!runtime.load("./hunyuan_ocr_ncnn_model", options, &error)) return 1;

hunyuan_ocr::InferenceRequest request;
request.prompt_mode = hunyuan_ocr::PromptMode::Document;
request.max_tokens = 128;

hunyuan_ocr::InferenceResult result;
if (!runtime.infer_file("document.png", request, &result, &error)) return 2;
```

调用方已经持有解码后像素时，可以使用接收连续 RGB 字节的 `infer_rgb`。同一个
Runtime 按顺序处理请求，不保证并发调用安全；需要并发时可创建多个实例，但会占用
多份模型内存。每次请求结束后，图片张量、vision features 和 KV cache 都会释放。

## JSONL 批量推理

批量模式只加载一次模型，并为每个物理输入行按顺序写出一条结果：

```json
{"id":"page-1","image":"images/page-1.png","prompt_mode":"document","max_tokens":256}
{"id":"page-2","image":"images/page-2.png","prompt":"只输出发票号码"}
```

```bash
./build/hunyuan_ocr_cli \
  --model ./hunyuan_ocr_ncnn_model \
  --batch-input requests.jsonl \
  --batch-output results.jsonl \
  --num-threads 8
```

`id` 必须非空且唯一。每条记录必须在 `prompt_mode`（`spotting` 或 `document`）
与 `prompt` 中二选一，`max_tokens` 可以省略。相对图片路径以输入 JSONL 所在目录
为基准。无效或推理失败的记录会写成 `ok: false`，后续记录仍继续执行；只要存在
失败记录，进程最终返回非零退出码。输出文件已存在时默认拒绝覆盖，显式传入
`--force` 才会重写。

## 运行示例

查看内置示例图：

```bash
python tools/run_example.py --list
```

运行单张图片：

```bash
python tools/run_example.py \
  --model ./hunyuan_ocr_ncnn_model \
  --case hf_demo
```

运行单张图片并传入自定义 prompt：

```bash
python tools/run_example.py \
  --model ./hunyuan_ocr_ncnn_model \
  --case hf_demo \
  --prompt "只输出图片中的可见文字"
```

运行全部示例图：

```bash
python tools/run_examples.py \
  --model ./hunyuan_ocr_ncnn_model
```

示例图片位于 `examples/images/`，来源记录在 `examples/IMAGE_SOURCES.md`。

## 完整测试

准备好 baseline/export 产生的 fixture 后，可以运行完整 token/text 测试：

```bash
python tools/run_regression.py \
  --package \
  --package-vision-backend dynamic
```

期望摘要：

```text
summary: 28/28 passed
```

该测试会比较 prompt ids、position ids、generated token ids 和最终 decode 文本。

## 性能测试

构建 CLI 并准备好打包模型后：

```bash
python tools/benchmark.py \
  --model ./hunyuan_ocr_ncnn_model \
  --cases hf_demo \
  --repeat 3 \
  --warmup 1 \
  --max-tokens 64
```

输出包含图片预处理、vision、prompt 组装、text embedding、prefill、增量 decode、总耗时和 decode token/s。更多说明见 `tools/README.md#benchmark`。

## 当前限制

- dynamic vision backend 已在 28 个 case 的前 128 个生成 token 上完成 token/text 测试，验证口径为 `max_pixels=524288`。
- HunyuanOCR 1.5 仍是绑定固定 checkpoint revision 和已验证 128-token 窗口的 preview。
- 有损 JPEG 的跨解码器舍入差异可能改变决策敏感样本的生成 token；严格对齐时使用规范化无损输入。
- 自定义 prompt 已接入 C++ tokenizer encode；更多 tokenizer 边界还需要继续补充 HF 对齐测试。
- 当前交付范围使用 `max_pixels=524288`，不包含原版高分辨率路径。
- 公开示例脚本只验证端到端运行；严格 token/text 对齐由 fixture 测试验证。

## 许可证

本仓库原创代码使用 Apache-2.0。`third_party/stb_image.h` 保留上游 MIT/Public Domain
许可说明，vendored picojson 保留 BSD 2-Clause 许可说明。

HunyuanOCR 模型文件遵循 Tencent Hunyuan Community License Agreement。
