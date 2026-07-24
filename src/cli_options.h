#pragma once

#include "hunyuan_ocr/vision_runtime.h"

#include <string>

namespace hunyuan_ocr::cli {

constexpr int kDefaultImageMaxTokens = 8192;

struct CliOptions {
    std::string model_root;
    std::string vlm_fixture_dir;
    bool dflash = false;
    bool mmap_weights = false;
    bool interactive = false;
    bool vulkan = false;
    int vulkan_device = 0;
    std::string image_path;
    std::string batch_input_path;
    std::string batch_output_path;
    bool force_batch_output = false;
    std::string prompt_mode_text;
    std::string prompt_text;
    int max_tokens = 0;
    float repetition_penalty = 1.08f;
    bool benchmark = false;
    int benchmark_warmup = 0;
    int benchmark_repeat = 1;
    int num_threads = 0;
    bool vision_vulkan = false;
    int vision_vulkan_device = 0;
    bool text_vulkan = false;
    int text_vulkan_device = 0;

    bool batch_mode = false;
    bool plain_image_inference = false;
    VisionRuntimeOptions vision_options;
};

enum class CliParseResult {
    Run,
    ExitSuccess,
    ExitFailure,
};

CliParseResult parse_cli_options(int argc, char** argv, CliOptions* options);

} // namespace hunyuan_ocr::cli
