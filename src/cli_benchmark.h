#pragma once

#include "hunyuan_ocr/vision_runtime.h"

#include <chrono>
#include <string>

namespace hunyuan_ocr::cli {

int run_image_benchmark(const std::string& model_root,
                        const std::string& image_path,
                        const std::string& prompt_mode_text,
                        const std::string& prompt_text,
                        int max_tokens,
                        float repetition_penalty,
                        int num_threads,
                        int warmup,
                        int repeat,
                        const VisionRuntimeOptions& vision_options,
                        bool text_vulkan,
                        int text_vulkan_device,
                        std::chrono::steady_clock::time_point process_start);

} // namespace hunyuan_ocr::cli
