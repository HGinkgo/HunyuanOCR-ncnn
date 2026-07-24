#include "cli_benchmark.h"

#include "cli_fixture.h"

#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/image_preprocessor.h"
#include "hunyuan_ocr/prompt_builder.h"
#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/tokenizer.h"
#include "hunyuan_ocr/vision_runtime.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace hunyuan_ocr::cli {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_token_vector(const char* title, const std::vector<int>& tokens)
{
    std::cout << title;
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        if (i != 0) std::cout << ' ';
        std::cout << tokens[i];
    }
    std::cout << "\n";
}

struct BenchmarkTiming {
    double preprocess_ms = 0.0;
    double vision_infer_ms = 0.0;
    double prompt_ms = 0.0;
    double text_embed_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double lm_head_ms = 0.0;
    double token_select_ms = 0.0;
    double tokenizer_decode_ms = 0.0;
    double text_total_ms = 0.0;
    double warm_inference_total_ms = 0.0;
};

struct BenchmarkIteration {
    BenchmarkTiming timing;
    std::vector<int> generated_tokens;
    std::string generated_text;
};

void add_benchmark_timing(BenchmarkTiming* total, const BenchmarkTiming& value)
{
    total->preprocess_ms += value.preprocess_ms;
    total->vision_infer_ms += value.vision_infer_ms;
    total->prompt_ms += value.prompt_ms;
    total->text_embed_ms += value.text_embed_ms;
    total->prefill_ms += value.prefill_ms;
    total->decode_ms += value.decode_ms;
    total->lm_head_ms += value.lm_head_ms;
    total->token_select_ms += value.token_select_ms;
    total->tokenizer_decode_ms += value.tokenizer_decode_ms;
    total->text_total_ms += value.text_total_ms;
    total->warm_inference_total_ms += value.warm_inference_total_ms;
}

void divide_benchmark_timing(BenchmarkTiming* value, double divisor)
{
    value->preprocess_ms /= divisor;
    value->vision_infer_ms /= divisor;
    value->prompt_ms /= divisor;
    value->text_embed_ms /= divisor;
    value->prefill_ms /= divisor;
    value->decode_ms /= divisor;
    value->lm_head_ms /= divisor;
    value->token_select_ms /= divisor;
    value->tokenizer_decode_ms /= divisor;
    value->text_total_ms /= divisor;
    value->warm_inference_total_ms /= divisor;
}

bool build_benchmark_prompt(const std::string& prompt_mode_text,
                            const std::string& prompt_text,
                            const hunyuan_ocr::Tokenizer& tokenizer,
                            const hunyuan_ocr::ImagePreprocessResult& image,
                            int merge_size,
                            hunyuan_ocr::PromptBuildResult* prompt,
                            std::string* error)
{
    if (!prompt_text.empty())
    {
        const std::vector<int> prompt_token_ids = tokenizer.encode(prompt_text, error);
        if (error && !error->empty())
        {
            return false;
        }
        return hunyuan_ocr::build_hunyuan_ocr_prompt_from_tokens(prompt_token_ids,
                                                                 image.grid_h,
                                                                 image.grid_w,
                                                                 merge_size,
                                                                 prompt,
                                                                 error);
    }

    hunyuan_ocr::PromptMode prompt_mode = hunyuan_ocr::PromptMode::Spotting;
    if (!hunyuan_ocr::parse_prompt_mode(prompt_mode_text, &prompt_mode, error))
    {
        return false;
    }
    return hunyuan_ocr::build_hunyuan_ocr_prompt(prompt_mode,
                                                 image.grid_h,
                                                 image.grid_w,
                                                 merge_size,
                                                 prompt,
                                                 error);
}

bool run_benchmark_iteration(const std::string& image_path,
                             const std::string& prompt_mode_text,
                             const std::string& prompt_text,
                             int max_tokens,
                             float repetition_penalty,
                             const hunyuan_ocr::ImagePreprocessor& preprocessor,
                             const hunyuan_ocr::VisionRuntime& vision_runtime,
                             const hunyuan_ocr::TextRuntime& text_runtime,
                             const hunyuan_ocr::Tokenizer& tokenizer,
                             const hunyuan_ocr::ImagePreprocessResult* prepared_image,
                             double prepared_preprocess_ms,
                             BenchmarkIteration* result,
                             std::string* error)
{
    const auto total_start = Clock::now();
    BenchmarkIteration local;

    hunyuan_ocr::ImagePreprocessResult image_storage;
    const hunyuan_ocr::ImagePreprocessResult* image = prepared_image;
    if (image == nullptr)
    {
        const auto preprocess_start = Clock::now();
        if (!preprocessor.preprocess_image_file(image_path, &image_storage, error))
        {
            return false;
        }
        local.timing.preprocess_ms = elapsed_ms(preprocess_start, Clock::now());
        image = &image_storage;
    }
    else
    {
        local.timing.preprocess_ms = prepared_preprocess_ms;
    }

    hunyuan_ocr::VisionRuntimeResult vision;
    const auto vision_start = Clock::now();
    if (!vision_runtime.run_dynamic_pixel_values(image->pixel_values,
                                                 image->grid_h,
                                                 image->grid_w,
                                                 preprocessor.config().merge_size,
                                                 &vision,
                                                 error))
    {
        return false;
    }
    local.timing.vision_infer_ms = elapsed_ms(vision_start, Clock::now());

    hunyuan_ocr::PromptBuildResult prompt;
    const auto prompt_start = Clock::now();
    if (!build_benchmark_prompt(prompt_mode_text,
                                prompt_text,
                                tokenizer,
                                *image,
                                preprocessor.config().merge_size,
                                &prompt,
                                error))
    {
        return false;
    }
    local.timing.prompt_ms = elapsed_ms(prompt_start, Clock::now());

    hunyuan_ocr::TextDecodeResult decode;
    if (!text_runtime.run_vlm_decode_with_prompt(prompt.input_ids,
                                                 prompt.position_ids,
                                                 prompt.image_token_id,
                                                 vision.vision_features,
                                                 vision.vision_token_count,
                                                 {},
                                                 max_tokens,
                                                 repetition_penalty,
                                                 &decode,
                                                 error))
    {
        return false;
    }
    local.timing.text_embed_ms = decode.timing.text_embed_ms;
    local.timing.prefill_ms = decode.timing.prefill_ms;
    local.timing.decode_ms = decode.timing.decode_ms;
    local.timing.lm_head_ms = decode.timing.lm_head_ms;
    local.timing.token_select_ms = decode.timing.token_select_ms;
    local.timing.text_total_ms = decode.timing.total_ms;
    local.generated_tokens = decode.generated_tokens;

    const auto tokenizer_decode_start = Clock::now();
    local.generated_text = tokenizer.decode(local.generated_tokens, true);
    local.timing.tokenizer_decode_ms = elapsed_ms(tokenizer_decode_start, Clock::now());
    local.timing.warm_inference_total_ms = elapsed_ms(total_start, Clock::now());

    *result = std::move(local);
    return true;
}

} // namespace

int run_image_benchmark(const std::string& model_root,
                        const std::string& image_path,
                        const std::string& prompt_mode_text,
                        const std::string& prompt_text,
                        int max_tokens,
                        float repetition_penalty,
                        int num_threads,
                        int warmup,
                        int repeat,
                        const hunyuan_ocr::VisionRuntimeOptions& vision_options,
                        bool text_vulkan,
                        int text_vulkan_device,
                        Clock::time_point process_start)
{
    std::string error;
    hunyuan_ocr::ImagePreprocessor preprocessor;
    hunyuan_ocr::ImagePreprocessResult probe_image;
    const auto probe_preprocess_start = Clock::now();
    if (!preprocessor.preprocess_image_file(image_path, &probe_image, &error))
    {
        std::cerr << "Benchmark preprocess failed: " << error << "\n";
        return 1;
    }
    const double probe_preprocess_ms = elapsed_ms(probe_preprocess_start, Clock::now());

    std::string resolved_vision_param_path;
    std::string resolved_vision_bin_path;
    std::string resolved_pos_embed_path;
    if (!resolve_dynamic_vision_paths(model_root,
                                      resolved_vision_param_path,
                                      resolved_vision_bin_path,
                                      resolved_pos_embed_path))
    {
        std::cerr << "Benchmark could not resolve the canonical dynamic vision model\n";
        return 1;
    }

    hunyuan_ocr::VisionRuntime vision_runtime(vision_options);
    const auto vision_load_start = Clock::now();
    if (!vision_runtime.load_dynamic(resolved_vision_param_path,
                                     resolved_vision_bin_path,
                                     resolved_pos_embed_path,
                                     &error))
    {
        std::cerr << "Benchmark vision load failed: " << error << "\n";
        return 1;
    }
    print_vision_compute_backend(vision_options, vision_runtime);
    const double vision_load_ms = elapsed_ms(vision_load_start, Clock::now());

    hunyuan_ocr::TextRuntime text_runtime(num_threads,
                                          vision_options.mmap_weights,
                                          text_vulkan,
                                          text_vulkan_device);
    const auto text_load_start = Clock::now();
    if (!text_runtime.load(model_root, &error))
    {
        std::cerr << "Benchmark text load failed: " << error << "\n";
        return 1;
    }
    const double text_load_ms = elapsed_ms(text_load_start, Clock::now());

    hunyuan_ocr::Tokenizer tokenizer;
    const auto tokenizer_load_start = Clock::now();
    if (!tokenizer.load(model_root + "/tokenizer/vocab.txt",
                        model_root + "/tokenizer/merges.txt",
                        model_root + "/tokenizer/special_tokens.json",
                        &error))
    {
        std::cerr << "Benchmark tokenizer load failed: " << error << "\n";
        return 1;
    }
    const double tokenizer_load_ms = elapsed_ms(tokenizer_load_start, Clock::now());
    const size_t mapped_weight_bytes =
        vision_runtime.mapped_weight_bytes() + text_runtime.mapped_weight_bytes();

    BenchmarkIteration cold;
    if (!run_benchmark_iteration(image_path,
                                 prompt_mode_text,
                                 prompt_text,
                                 max_tokens,
                                 repetition_penalty,
                                 preprocessor,
                                 vision_runtime,
                                 text_runtime,
                                 tokenizer,
                                 &probe_image,
                                 probe_preprocess_ms,
                                 &cold,
                                 &error))
    {
        std::cerr << "Benchmark cold inference failed: " << error << "\n";
        return 1;
    }
    const double cold_start_total_ms = elapsed_ms(process_start, Clock::now());

    for (int i = 0; i < warmup; ++i)
    {
        BenchmarkIteration iteration;
        if (!run_benchmark_iteration(image_path,
                                     prompt_mode_text,
                                     prompt_text,
                                     max_tokens,
                                     repetition_penalty,
                                     preprocessor,
                                     vision_runtime,
                                     text_runtime,
                                     tokenizer,
                                     nullptr,
                                     0.0,
                                     &iteration,
                                     &error))
        {
            std::cerr << "Benchmark warmup failed: " << error << "\n";
            return 1;
        }
        if (iteration.generated_tokens != cold.generated_tokens ||
            iteration.generated_text != cold.generated_text)
        {
            std::cerr << "Benchmark warmup output differs from cold inference\n";
            return 3;
        }
    }

    BenchmarkTiming average;
    for (int i = 0; i < repeat; ++i)
    {
        BenchmarkIteration iteration;
        if (!run_benchmark_iteration(image_path,
                                     prompt_mode_text,
                                     prompt_text,
                                     max_tokens,
                                     repetition_penalty,
                                     preprocessor,
                                     vision_runtime,
                                     text_runtime,
                                     tokenizer,
                                     nullptr,
                                     0.0,
                                     &iteration,
                                     &error))
        {
            std::cerr << "Benchmark measured inference failed: " << error << "\n";
            return 1;
        }
        if (iteration.generated_tokens != cold.generated_tokens ||
            iteration.generated_text != cold.generated_text)
        {
            std::cerr << "Benchmark measured output differs from cold inference\n";
            return 3;
        }
        add_benchmark_timing(&average, iteration.timing);
    }
    divide_benchmark_timing(&average, static_cast<double>(repeat));

    const size_t generated_token_count = cold.generated_tokens.size();
    const double generated_per_s = average.warm_inference_total_ms > 0.0
        ? static_cast<double>(generated_token_count) * 1000.0 / average.warm_inference_total_ms
        : 0.0;
    const int decode_steps = generated_token_count > 0
        ? static_cast<int>(generated_token_count) - 1
        : 0;
    const double decode_token_per_s = average.decode_ms > 0.0
        ? static_cast<double>(decode_steps) * 1000.0 / average.decode_ms
        : 0.0;
    const int effective_threads = num_threads > 0
        ? num_threads
        : hunyuan_ocr::make_fp32_ncnn_option().num_threads;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Benchmark:\n";
    std::cout << "  num_threads: " << effective_threads << "\n";
    std::cout << "  warmup: " << warmup << "\n";
    std::cout << "  repeat: " << repeat << "\n";
    std::cout << "  mmap_weights: " << (vision_options.mmap_weights ? 1 : 0) << "\n";
    std::cout << "  mapped_weight_bytes: " << mapped_weight_bytes << "\n";
    std::cout << "  cold_start_total_ms: " << cold_start_total_ms << "\n";
    std::cout << "  vision_load_ms: " << vision_load_ms << "\n";
    std::cout << "  text_load_ms: " << text_load_ms << "\n";
    std::cout << "  tokenizer_load_ms: " << tokenizer_load_ms << "\n";
    std::cout << "  preprocess_ms: " << average.preprocess_ms << "\n";
    std::cout << "  vision_infer_ms: " << average.vision_infer_ms << "\n";
    std::cout << "  prompt_ms: " << average.prompt_ms << "\n";
    std::cout << "  text_embed_ms: " << average.text_embed_ms << "\n";
    std::cout << "  prefill_ms: " << average.prefill_ms << "\n";
    std::cout << "  decode_ms: " << average.decode_ms << "\n";
    std::cout << "  lm_head_ms: " << average.lm_head_ms << "\n";
    std::cout << "  token_select_ms: " << average.token_select_ms << "\n";
    std::cout << "  tokenizer_decode_ms: " << average.tokenizer_decode_ms << "\n";
    std::cout << "  text_total_ms: " << average.text_total_ms << "\n";
    std::cout << "  warm_inference_total_ms: " << average.warm_inference_total_ms << "\n";
    std::cout << "  generated_token_count: " << generated_token_count << "\n";
    std::cout << "  generated_token_per_s: " << generated_per_s << "\n";
    std::cout << "  decode_token_per_s: " << decode_token_per_s << "\n";
    print_token_vector("Benchmark generated tokens: ", cold.generated_tokens);
    return 0;
}


} // namespace hunyuan_ocr::cli
