# Reusable Runtime And JSONL Batch Design

## Goal

Turn the existing `HunyuanOCR` public class into the owner of a reusable,
long-lived OCR runtime and add sequential JSONL batch inference to the existing
CLI. Model networks load once, while image data, vision features, prompt state,
and KV caches remain request-scoped.

## Scope

- Extend `HunyuanOCR` with PIMPL-backed model ownership, `infer_file`, and
  contiguous-RGB `infer_rgb` entry points.
- Keep the public API free of `ncnn::Mat`, Extractor, and KV-cache types.
- Return complete text, token IDs, timings, decoder mode, and optional DFlash
  statistics after inference. Streaming callbacks are out of scope.
- Add `--batch-input`, `--batch-output`, and `--force` to
  `hunyuan_ocr_cli`; batch mode and single-image mode are mutually exclusive.
- Keep build-tree use through the existing `hunyuan_ocr` static target.
  Installed CMake packages and ABI stability are out of scope.

## Runtime API

`RuntimeOptions` owns process-wide choices: thread count, vision Vulkan device,
AR versus DFlash, and repetition penalty. `InferenceRequest` owns per-request
choices: spotting/document/custom prompt and `max_tokens`. A custom request must
contain non-empty UTF-8 prompt text.

`HunyuanOCR::load` validates the model layout and loads the tokenizer, vision
runtime, text runtime, and optional DFlash network. The existing one-argument
load overload remains as a convenience wrapper using default options, but its
meaning becomes full runtime loading. Failed loads leave the object not ready.

One instance supports one inference call at a time and is not thread-safe.
Instances are movable but not copyable. Callers that require concurrency create
independent instances and accept the duplicated model memory.

## Inference Flow

Both image entry points converge after image decoding:

1. Validate runtime readiness and request options.
2. Preprocess the original RGB image using the packaged processor limits.
3. Run the selected dynamic vision graph or fixed-grid fallback.
4. Tokenize and build the multimodal prompt.
5. Run AR or DFlash text generation.
6. Decode token IDs, assemble timing/statistics, and return.

The implementation reuses existing `ImagePreprocessor`, `VisionRuntime`,
`TextRuntime`, `Tokenizer`, and prompt-building code. It does not introduce a
runtime cleanup API: request-local values leave scope after every call, while
the networks and tokenizer remain resident.

## JSONL Contract

Each physical input line must be one non-empty JSON object with a required,
unique non-empty `id`, required `image`, exactly one of `prompt_mode` or
`prompt`, and optional positive integer `max_tokens`. `prompt_mode` accepts
`spotting` or `document`. Relative image paths resolve from the input JSONL
directory. Runtime and decoder settings cannot vary per line.

The CLI writes and flushes one output object per physical input line, preserving
order. Every output contains the one-based input `line`. Successful records
also contain `id`, original `image`, `ok`, decoder, decoded text, token IDs,
generated-token count, and preprocess/vision/text/total timings. DFlash records
also contain block, drafted-token, and accepted-token counts. Failure records
contain `id` and image when they could be parsed, `ok: false`, an error stage,
and message. Empty, malformed, invalid, or failed records do not stop later
records. Any failed record makes the final process exit nonzero.

An existing output path is rejected unless `--force` is supplied. Force mode
truncates the output before processing. Automatic resume and result merging are
out of scope.

JSON parsing and serialization use a pinned vendored `picojson.h` rather than
ad hoc parsing or a system dependency. Its BSD-2-Clause notice is retained.

## Compatibility And Errors

The existing single-image CLI, fixtures, benchmark, UTF-8 Windows arguments,
CPU defaults, Vulkan opt-in, and DFlash opt-in remain behaviorally compatible.
Ordinary API failures use `RuntimeError { stage, message }`, not exceptions.
Stages identify model layout/load, request validation, image decode/preprocess,
vision, prompt/tokenizer, text generation, and batch input/output.

## Verification

- Model-free API tests cover defaults, move/copy traits, invalid state, request
  validation, and RGB size validation using only public headers.
- Model-free JSONL tests cover schema validation, escaping, Unicode, duplicate
  IDs, relative paths, ordered continue-on-error behavior, flushing-compatible
  output, existing-output rejection, and force overwrite.
- Existing CLI tests cover option presence and mode conflicts on Linux and
  Windows; CI builds the new tests with MSVC and UCRT64.
- Full Release CTest and ASAN/UBSAN CTest must pass.
- Model-backed validation compares existing single-image and new same-process
  JSONL outputs for representative dynamic sizes, custom prompt, AR, and
  DFlash. It records cold load, per-item timings, total wall time, and RSS.
