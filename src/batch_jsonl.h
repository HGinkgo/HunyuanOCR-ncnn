#pragma once

#include "hunyuan_ocr/hunyuan_ocr.h"

#include <cstddef>
#include <functional>
#include <string>

namespace hunyuan_ocr {
namespace detail {

struct BatchOptions {
    std::string input_path;
    std::string output_path;
    bool force = false;
    int default_max_tokens = 128;
};

struct BatchRequest {
    size_t line = 0;
    std::string id;
    std::string image;
    std::string resolved_image;
    InferenceRequest request;
};

struct BatchSummary {
    size_t total = 0;
    size_t succeeded = 0;
    size_t failed = 0;
};

using BatchInfer = std::function<bool(const BatchRequest& request,
                                      InferenceResult* result,
                                      RuntimeError* error)>;

bool run_jsonl_batch(const BatchOptions& options,
                     const BatchInfer& infer,
                     BatchSummary* summary,
                     RuntimeError* error);

} // namespace detail
} // namespace hunyuan_ocr
