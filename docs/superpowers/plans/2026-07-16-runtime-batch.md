# Reusable Runtime And JSONL Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a reusable, long-lived HunyuanOCR C++ runtime and ordered, fault-tolerant JSONL batch inference in the existing CLI.

**Architecture:** Upgrade the existing `HunyuanOCR` facade to a move-only PIMPL owner that composes the already-tested preprocessing, vision, tokenizer, prompt, text, and DFlash modules. Keep JSONL parsing and batch execution in an internal component with an injectable inference callback so all control-flow behavior is tested without model weights.

**Tech Stack:** C++17, CMake, ncnn, stb_image, vendored picojson, CTest, Python CLI contract tests, ASAN/UBSAN.

---

## File Map

- Modify `include/hunyuan_ocr/hunyuan_ocr.h`: public runtime options, request,
  result, error, and move-only runtime API.
- Modify `src/hunyuan_ocr.cpp`: retain version/options helpers and implement the
  PIMPL-backed high-level runtime orchestration.
- Create `src/batch_jsonl.h`: internal batch records, options, and injectable
  runner declarations.
- Create `src/batch_jsonl.cpp`: JSONL parsing, serialization, validation, path
  resolution, output policy, and ordered execution.
- Vendor `third_party/picojson.h` and `third_party/picojson-LICENSE`: pinned JSON
  parser and license.
- Modify `src/main.cpp`: parse batch flags, share runtime construction, dispatch
  single-image and batch inference through `HunyuanOCR`.
- Create `tests/hunyuan_ocr_api_test.cpp`: public-header and model-free runtime
  contract tests.
- Create `tests/batch_jsonl_test.cpp`: model-free JSONL codec and execution tests.
- Modify `tests/cli_options_test.py`, `tests/windows_workflow_test.py`, and
  `.github/workflows/windows-compile.yml`: CLI and Windows gates.
- Modify `CMakeLists.txt`: compile the batch component and register tests.
- Modify `README.md`, `README_zh.md`, and `NOTICE`: concise API/batch usage and
  vendored dependency attribution.

### Task 1: Public Runtime Contract

**Files:**
- Modify: `include/hunyuan_ocr/hunyuan_ocr.h`
- Modify: `src/hunyuan_ocr.cpp`
- Create: `tests/hunyuan_ocr_api_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the public-contract red test**

Add compile-time assertions that `HunyuanOCR` is move constructible and not
copy constructible. Exercise default `RuntimeOptions`, `InferenceRequest`, and
`RuntimeError`, then verify inference before load fails with stage
`runtime_state`, invalid custom prompts fail with `request`, and an RGB vector
whose size differs from `width * height * 3` fails with `image_input`.

```cpp
static_assert(!std::is_copy_constructible<hunyuan_ocr::HunyuanOCR>::value, "runtime must not copy");
static_assert(std::is_move_constructible<hunyuan_ocr::HunyuanOCR>::value, "runtime must move");

hunyuan_ocr::HunyuanOCR runtime;
hunyuan_ocr::InferenceRequest request;
hunyuan_ocr::InferenceResult result;
hunyuan_ocr::RuntimeError error;
if (runtime.infer_file("missing.png", request, &result, &error) ||
    error.stage != "runtime_state") return 1;
```

- [ ] **Step 2: Register and run the red test**

Add `hunyuan_ocr_api_test` to CMake, link only the public `hunyuan_ocr` target,
and register CTest name `hunyuan_ocr_api`.

Run:

```bash
cmake --build build-baseline -j2
```

Expected: compilation fails because the new public types and methods do not
exist.

- [ ] **Step 3: Add the minimal public types and move-only PIMPL shell**

Define `RuntimeOptions`, `PromptMode`, `InferenceRequest`, `InferenceTiming`,
`DecoderStatistics`, `InferenceResult`, and `RuntimeError`. Give `HunyuanOCR`
an out-of-line destructor, move constructor/assignment, deleted copies, a
`std::unique_ptr<Impl>`, the three-argument `load`, convenience `load`, and two
inference methods.

```cpp
class HunyuanOCR {
public:
    HunyuanOCR();
    ~HunyuanOCR();
    HunyuanOCR(HunyuanOCR&&) noexcept;
    HunyuanOCR& operator=(HunyuanOCR&&) noexcept;
    HunyuanOCR(const HunyuanOCR&) = delete;
    HunyuanOCR& operator=(const HunyuanOCR&) = delete;
    // load/infer declarations
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

Validate state and inputs before touching model code. Clear the result and error
at method entry so failed calls cannot expose stale data.

- [ ] **Step 4: Run the focused green test**

Run:

```bash
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure -R '^hunyuan_ocr_api$'
```

Expected: one test passes.

- [ ] **Step 5: Commit the public contract**

```bash
git add include/hunyuan_ocr/hunyuan_ocr.h src/hunyuan_ocr.cpp \
  tests/hunyuan_ocr_api_test.cpp CMakeLists.txt
git commit -m "refactor: define reusable OCR runtime API"
```

### Task 2: Full Runtime Orchestration

**Files:**
- Modify: `src/hunyuan_ocr.cpp`
- Modify: `tests/hunyuan_ocr_api_test.cpp`

- [ ] **Step 1: Extend red tests for load and request validation**

Create a temporary incomplete model directory and assert `load` fails with
`model_layout`, leaves `ready()` false, and retains a layout report identifying
missing required files. Test zero dimensions, non-positive `max_tokens`, empty
custom prompt, and non-empty custom text in a non-custom mode.

- [ ] **Step 2: Run the red tests**

```bash
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure -R '^hunyuan_ocr_api$'
```

Expected: validation assertions fail until the implementation distinguishes
the required stages.

- [ ] **Step 3: Implement model ownership and shared inference flow**

Make `Impl` own `ImagePreprocessor`, `VisionRuntime`, `TextRuntime`, `Tokenizer`,
the chosen dynamic/fixed backend metadata, runtime options, and layout report.
Move the normal-image orchestration currently in `main.cpp` behind a shared
private function:

```cpp
bool infer_preprocessed(const ImagePreprocessResult& image,
                        const InferenceRequest& request,
                        InferenceResult* result,
                        RuntimeError* error);
```

Load dynamic vision when the package provides its root files, otherwise select
the exact fixed grid produced by preprocessing. Load DFlash only when requested
and report missing draft/auxiliary assets as `dflash_load`. Map lower-level
string errors to stable public stages. Compute total timing around the complete
request and preserve lower-level text/DFlash timing fields.

- [ ] **Step 4: Make both image entry points converge**

`infer_file` calls `preprocess_image_file`. `infer_rgb` validates a contiguous
RGB vector of exactly `width * height * 3` bytes with overflow checks and calls
`preprocess_original_rgb`. Both then call `infer_preprocessed`.

- [ ] **Step 5: Run API and existing runtime tests**

```bash
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure \
  -R '^(hunyuan_ocr_api|prompt_builder|vision_dynamic_runtime|dflash_runtime|memory_lifecycle|kv_cache_lifecycle)$'
```

Expected: six tests pass with no model weights required beyond the configured
dynamic-vision fixture used by the existing test.

- [ ] **Step 6: Commit runtime orchestration**

```bash
git add include/hunyuan_ocr/hunyuan_ocr.h src/hunyuan_ocr.cpp \
  tests/hunyuan_ocr_api_test.cpp
git commit -m "feat: add reusable OCR inference runtime"
```

### Task 3: Structured JSONL Codec And Batch Runner

**Files:**
- Create: `third_party/picojson.h`
- Create: `third_party/picojson-LICENSE`
- Create: `src/batch_jsonl.h`
- Create: `src/batch_jsonl.cpp`
- Create: `tests/batch_jsonl_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `NOTICE`

- [ ] **Step 1: Vendor and verify picojson**

Fetch upstream `picojson.h` at git blob
`76742fe06ac4f6cfc26c09a3445a1def071a5051` and its BSD-2-Clause license. Verify:

```bash
git hash-object third_party/picojson.h
```

Expected: `76742fe06ac4f6cfc26c09a3445a1def071a5051`.

- [ ] **Step 2: Write JSONL red tests**

Cover a valid document request, a UTF-8 custom prompt, escaped newlines and
quotes, blank-line rejection, malformed JSON, missing/empty/duplicate IDs,
missing image, both/neither prompt fields, invalid prompt mode, fractional or
non-positive `max_tokens`, and relative image resolution from the manifest
directory.

Add an injectable callback that fails the middle of three records and assert:

```cpp
if (seen_ids != std::vector<std::string>{"a", "b", "c"}) return 1;
if (summary.total != 3 || summary.succeeded != 2 || summary.failed != 1) return 2;
```

Verify three ordered output lines, complete success fields, structured failure,
output-exists rejection, and force truncation.

- [ ] **Step 3: Register and run the red test**

Add `src/batch_jsonl.cpp` to `hunyuan_ocr`, expose `src` only to the test, and
register CTest name `batch_jsonl`.

```bash
cmake --build build-baseline -j2
```

Expected: compilation fails because batch declarations and implementation are
missing.

- [ ] **Step 4: Implement strict parsing and serialization**

Use `picojson::parse` for each physical line and require a JSON object. Reject
unknown fields to catch misspellings. Validate exact JSON types and integer
range before conversion. Use picojson values for output serialization so text,
paths, and errors are escaped correctly. Do not concatenate JSON strings.

Define an internal runner callback:

```cpp
using BatchInfer = std::function<bool(const BatchRequest&,
                                      InferenceResult*, RuntimeError*)>;
```

Open the output with exclusive-by-default semantics, write one serialized line
per parsed input record, flush after each line, continue after record/inference
failures, and return a summary. Treat input/output open or write failures as
fatal batch errors; record-level parse/schema/inference failures count toward a
nonzero final status.

- [ ] **Step 5: Run focused green tests and diff checks**

```bash
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure \
  -R '^(batch_jsonl|hunyuan_ocr_api|utf8|utf8_cpp20)$'
git diff --check
```

Expected: four tests pass and diff check is clean.

- [ ] **Step 6: Commit JSONL support**

```bash
git add third_party/picojson.h third_party/picojson-LICENSE NOTICE \
  src/batch_jsonl.h src/batch_jsonl.cpp tests/batch_jsonl_test.cpp CMakeLists.txt
git commit -m "feat: add JSONL batch execution core"
```

### Task 4: CLI Migration And Batch Mode

**Files:**
- Modify: `src/main.cpp`
- Modify: `tests/cli_options_test.py`
- Modify: `tests/windows_workflow_test.py`
- Modify: `.github/workflows/windows-compile.yml`

- [ ] **Step 1: Add CLI red tests**

Require help text for `--batch-input FILE`, `--batch-output FILE`, and `--force`.
Assert parser rejection for only one batch path, batch with `--image`, `--force`
outside batch mode, batch with fixture/benchmark/decode modes, and per-image
`--prompt` or `--prompt-mode` in batch mode.

```python
expect_failure(["--model", ".", "--batch-input", "in.jsonl"], "--batch-output")
expect_failure(["--model", ".", "--batch-input", "in.jsonl", "--batch-output", "out.jsonl", "--image", "x.png"], "mutually exclusive")
```

- [ ] **Step 2: Run the red CLI test**

```bash
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure -R '^cli_options$'
```

Expected: failure because batch flags are unknown.

- [ ] **Step 3: Parse and validate batch mode**

Add the three flags to the existing UTF-8 argument path. Centralize mode
validation after parsing and before model load. Preserve all existing diagnostic
fixture and benchmark combinations.

- [ ] **Step 4: Route normal image and batch inference through `HunyuanOCR`**

Construct one `RuntimeOptions` from CLI flags and load one `HunyuanOCR` object.
For single-image mode, translate CLI prompt fields to `InferenceRequest`, call
`infer_file`, and preserve the existing `Decoded text:` and token/timing output.
For batch mode, call the internal batch runner with a lambda that invokes the
same loaded runtime. Return zero only when every record succeeds.

- [ ] **Step 5: Extend Windows gates**

Add `hunyuan_ocr_api|batch_jsonl` to both MSVC and UCRT64 CTest filters and make
`windows_workflow_test.py` assert both names are present.

- [ ] **Step 6: Run focused CLI and workflow tests**

```bash
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure \
  -R '^(cli_options|batch_jsonl|hunyuan_ocr_api|windows_workflow|linux_workflow)$'
```

Expected: five tests pass.

- [ ] **Step 7: Commit CLI integration**

```bash
git add src/main.cpp tests/cli_options_test.py tests/windows_workflow_test.py \
  .github/workflows/windows-compile.yml
git commit -m "feat: add persistent JSONL batch inference"
```

### Task 5: User-Facing Usage And Complete Local Gates

**Files:**
- Modify: `README.md`
- Modify: `README_zh.md`
- Modify: `tests/release_metadata_test.py`

- [ ] **Step 1: Add concise API and JSONL examples**

Document `add_subdirectory`, linking `hunyuan_ocr`, one `infer_file` example,
the exact input/output JSONL fields, relative-path behavior, continue-on-error
exit status, and `--force`. State sequential/non-thread-safe semantics and that
model weights remain loaded between records.

- [ ] **Step 2: Extend metadata checks**

Assert both READMEs mention `--batch-input`, `--batch-output`, and
`infer_file`, and that NOTICE attributes picojson.

- [ ] **Step 3: Run complete Release verification**

```bash
cmake --build build-baseline -j2
ctest --test-dir build-baseline --output-on-failure
python3 tests/windows_workflow_test.py
git diff --check
```

Expected: all registered tests pass, the static workflow test exits zero, and
diff check prints nothing.

- [ ] **Step 4: Commit usage updates**

```bash
git add README.md README_zh.md NOTICE tests/release_metadata_test.py
git commit -m "docs: document reusable and batch inference"
```

### Task 6: Sanitizers And Model-Backed Acceptance

**Files:**
- No tracked source changes expected unless verification reveals a defect.

- [ ] **Step 1: Configure a separate sanitizer build with pinned paths**

```bash
cmake -S . -B build-sanitizers \
  -DCMAKE_BUILD_TYPE=Debug \
  -DHUNYUAN_OCR_BUILD_TESTS=ON \
  -DHUNYUAN_OCR_USE_NCNN_PACKAGE=OFF \
  -DNCNN_INCLUDE_DIR=/root/hpf/workspace/ncnn_hunyuanocr/ncnn/src \
  -DNCNN_BUILD_INCLUDE_DIR=/root/hpf/workspace/ncnn_hunyuanocr/ncnn/build/src \
  -DNCNN_LIBRARY=/root/hpf/workspace/ncnn_hunyuanocr/ncnn/build/src/libncnn.a \
  -DPython3_EXECUTABLE=/root/miniconda3/envs/HunyuanOCR-ncnn/bin/python \
  -DHUNYUAN_OCR_TEST_TOKENIZER_DIR=/root/hpf/workspace/ncnn_hunyuanocr/models/hunyuan_ocr_1_5/runtime/tokenizer \
  -DHUNYUAN_OCR_TEST_DYNAMIC_VISION_DIR=/root/hpf/workspace/ncnn_hunyuanocr/models/hunyuan_ocr_1_5/runtime/vision \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-sanitizers -j2
```

Expected: configure and build exit zero using the HunyuanOCR Conda interpreter.

- [ ] **Step 2: Run focused and full sanitizer gates**

```bash
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build-sanitizers --output-on-failure \
  -R '^(hunyuan_ocr_api|batch_jsonl|memory_lifecycle|kv_cache_lifecycle|precise_sdpa|dflash_runtime)$'

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build-sanitizers --output-on-failure
```

Expected: all selected and full tests pass without ASAN, LSAN, or UBSAN
diagnostics.

- [ ] **Step 3: Run representative model-backed JSONL comparisons**

Use `/root/hpf/workspace/ncnn_hunyuanocr/models/hunyuan_ocr_1_5/runtime` with
three canonical PNGs covering different dynamic sizes, document/spotting/custom
prompts, and compare each batch record's token IDs and text against a single
image invocation using the same options. Repeat once with DFlash for a packaged
case that has draft assets. Record model load time, item timings, total wall
time, and RSS before/after. Do not rerun the full 28-image suite unless a token
or text mismatch appears.

- [ ] **Step 4: Final repository checks**

```bash
git diff --check
git status --short
git log --oneline --decorate -8
```

Expected: only intentional tracked changes remain, no whitespace failures, and
all feature commits are on `p10-runtime-batch` without a push.
