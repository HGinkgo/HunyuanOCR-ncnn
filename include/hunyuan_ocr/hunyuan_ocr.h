#pragma once

#include "hunyuan_ocr/model_layout.h"
#include "hunyuan_ocr/prompt_builder.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ncnn {
class Option;
}

namespace hunyuan_ocr {

std::string project_version();
std::string ncnn_version();
ncnn::Option make_fp32_ncnn_option(int num_threads = 0);

struct RuntimeOptions {
    int num_threads = 0;
    bool vision_vulkan = false;
    int vision_vulkan_device = 0;
    bool dflash = false;
    float repetition_penalty = 1.08f;
    bool mmap_weights = false;
};

struct InferenceChunk {
    int token_id = -1;
    std::string text_delta;
};

using InferenceCallback = std::function<void(const InferenceChunk&)>;

struct InferenceRequest {
    PromptMode prompt_mode = PromptMode::Document;
    std::string prompt;
    int max_tokens = 128;
    // Invoked synchronously for each committed token; special tokens may have an empty text delta.
    InferenceCallback stream_callback;
};

struct InferenceTiming {
    double preprocess_ms = 0.0;
    double vision_ms = 0.0;
    double text_ms = 0.0;
    double total_ms = 0.0;
};

enum class DecoderMode {
    Autoregressive,
    DFlash,
};

struct DecoderStatistics {
    DecoderMode mode = DecoderMode::Autoregressive;
    int block_count = 0;
    int drafted_token_count = 0;
    int accepted_draft_token_count = 0;
};

struct InferenceResult {
    std::string text;
    std::vector<int> token_ids;
    InferenceTiming timing;
    DecoderStatistics decoder;
};

struct RuntimeError {
    std::string stage;
    std::string message;
};

class HunyuanOCR {
public:
    HunyuanOCR();
    ~HunyuanOCR();
    HunyuanOCR(HunyuanOCR&&) noexcept;
    HunyuanOCR& operator=(HunyuanOCR&&) noexcept;
    HunyuanOCR(const HunyuanOCR&) = delete;
    HunyuanOCR& operator=(const HunyuanOCR&) = delete;

    bool load(const std::string& model_root);
    bool load(const std::string& model_root,
              const RuntimeOptions& options,
              RuntimeError* error);
    bool infer_file(const std::string& image_path,
                    const InferenceRequest& request,
                    InferenceResult* result,
                    RuntimeError* error);
    bool infer_rgb(const std::vector<unsigned char>& rgb,
                   int width,
                   int height,
                   const InferenceRequest& request,
                   InferenceResult* result,
                   RuntimeError* error);
    bool ready() const;
    size_t mapped_weight_bytes() const;
    const ModelLayoutReport& layout_report() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hunyuan_ocr
