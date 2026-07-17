#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/image_preprocessor.h"
#include "hunyuan_ocr/prompt_builder.h"
#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/tokenizer.h"
#include "hunyuan_ocr/utf8.h"
#include "hunyuan_ocr/vision_runtime.h"

#include "batch_jsonl.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

hunyuan_ocr::VisionRuntimeOptions make_vision_runtime_options(
    int num_threads, bool use_vulkan, int vulkan_device, bool mmap_weights)
{
    hunyuan_ocr::VisionRuntimeOptions options;
    options.num_threads = num_threads;
    options.use_vulkan = use_vulkan;
    options.vulkan_device = vulkan_device;
    options.mmap_weights = mmap_weights;
    return options;
}

void print_vision_compute_backend(const hunyuan_ocr::VisionRuntimeOptions& options,
                                  const hunyuan_ocr::VisionRuntime& runtime)
{
    if (!options.use_vulkan) return;
    std::cout << "vision_backend: vulkan\n";
    std::cout << "vision_vulkan_device: " << options.vulkan_device << "\n";
    std::cout << "vision_gelu_cpu_fallback_count: "
              << runtime.gelu_cpu_fallback_count() << "\n";
}

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [--help] [--version] [--model PATH]\n"
        << "\n"
        << "HunyuanOCR-ncnn CLI for PNG/JPEG and JSONL inference.\n"
        << "\n"
        << "Options:\n"
        << "  --help          Show this help message.\n"
        << "  --version       Print project and ncnn version information.\n"
        << "  --model PATH    Check a HunyuanOCR ncnn model directory.\n"
        << "  --vlm-fixture PATH     Run input_ids + vision_features + text decode fixture.\n"
        << "  --dflash               Run DFlash with --vlm-fixture or --image with --prompt/--prompt-mode.\n"
        << "  --mmap-weights         Load model weights from read-only file mappings.\n"
        << "  --vision-vulkan       Run only the vision network with ncnn Vulkan fp32.\n"
        << "  --vision-vulkan-device N Vulkan device index for vision. Default: 0.\n"
        << "  --image PATH           Run PNG/JPEG image -> resize -> flattened pixel_values.\n"
        << "  --batch-input PATH     Read ordered inference requests from a JSONL file.\n"
        << "  --batch-output PATH    Write one ordered JSON result per input line.\n"
        << "  --force                Replace an existing --batch-output file.\n"
        << "  --prompt-mode MODE     Build built-in prompt tensors in C++: spotting or document.\n"
        << "  --prompt TEXT          Build a custom image prompt with the bundled tokenizer.\n"
        << "  --benchmark            Run cold-start and same-process warm inference timing.\n"
        << "  --benchmark-warmup N   Same-process warmup iterations. Default: 0.\n"
        << "  --benchmark-repeat N   Same-process measured iterations. Default: 1.\n"
        << "  --num-threads N        ncnn thread count for all submodels.\n"
        << "  --repetition-penalty F Greedy decode repetition penalty. Default: 1.08.\n"
        << "  --max-tokens N         Limit fixture generated token count.\n";
}

void print_file_group(const char* title, const std::vector<hunyuan_ocr::ModelFile>& files)
{
    std::cout << title << "\n";
    if (files.empty())
    {
        std::cout << "  (none)\n";
        return;
    }

    for (const auto& file : files)
    {
        std::cout << "  " << file.relative_path << " - " << file.note << "\n";
    }
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

bool read_text_file(const std::string& path, std::string& out)
{
    std::ifstream file(hunyuan_ocr::path_from_utf8(path));
    if (!file.is_open())
    {
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    out = buffer.str();
    return true;
}

bool file_exists(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

bool resolve_dynamic_vision_paths(const std::string& model_root,
                                  std::string& param_path,
                                  std::string& bin_path,
                                  std::string& pos_embed_path)
{
    const std::filesystem::path vision_dir = hunyuan_ocr::path_from_utf8(model_root) / "vision";
    const std::filesystem::path pos_path = vision_dir / "pos_embed.bin";
    const std::filesystem::path candidate_param = vision_dir / "vision.ncnn.param";
    const std::filesystem::path candidate_bin = vision_dir / "vision.ncnn.bin";
    if (file_exists(candidate_param) && file_exists(candidate_bin) && file_exists(pos_path))
    {
        param_path = hunyuan_ocr::path_to_utf8(candidate_param);
        bin_path = hunyuan_ocr::path_to_utf8(candidate_bin);
        pos_embed_path = hunyuan_ocr::path_to_utf8(pos_path);
        return true;
    }

    return false;
}

template <typename T>
bool read_binary_vector_file(const std::string& path, std::vector<T>& out, std::string& error)
{
    std::ifstream file(hunyuan_ocr::path_from_utf8(path), std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        error = "failed to open: " + path;
        return false;
    }
    const std::streamsize bytes = file.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0)
    {
        error = "invalid binary vector byte size: " + path;
        return false;
    }
    file.seekg(0, std::ios::beg);
    out.assign(static_cast<size_t>(bytes) / sizeof(T), T{});
    if (!out.empty())
    {
        file.read(reinterpret_cast<char*>(out.data()), bytes);
        if (file.gcount() != bytes)
        {
            error = "short read: " + path;
            return false;
        }
    }
    return true;
}

bool print_fixture_vision_diff(const std::vector<float>& actual,
                               const std::string& fixture_dir,
                               std::string& error)
{
    if (fixture_dir.empty()) return true;

    std::vector<float> expected;
    if (!read_binary_vector_file(fixture_dir + "/vision_features.f32", expected, error))
    {
        return false;
    }
    if (actual.size() != expected.size())
    {
        error = "vision feature size mismatch: got " + std::to_string(actual.size()) +
                ", expected " + std::to_string(expected.size());
        return false;
    }

    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const float diff = std::fabs(actual[i] - expected[i]);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
    }
    const double mean_abs = actual.empty() ? 0.0 : sum_abs / actual.size();
    std::cout << "vision_feature_max_abs_diff_fixture: " << max_abs << "\n";
    std::cout << "vision_feature_mean_abs_diff_fixture: " << mean_abs << "\n";
    return true;
}

std::string strip_trailing_newlines(std::string text)
{
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
    {
        text.pop_back();
    }
    return text;
}

int print_dflash_decode_result(const std::string& label,
                               const std::string& model_root,
                               const std::string& vlm_fixture_dir,
                               const hunyuan_ocr::DFlashDecodeResult& result)
{
    std::string error;
    hunyuan_ocr::Tokenizer tokenizer;
    if (!tokenizer.load(model_root + "/tokenizer/vocab.txt",
                        model_root + "/tokenizer/special_tokens.json",
                        &error))
    {
        std::cerr << "Failed to load tokenizer: " << error << "\n";
        return 1;
    }

    const hunyuan_ocr::TextDecodeResult& decode = result.decode;
    const std::string generated_text = tokenizer.decode(decode.generated_tokens, true);
    const auto safe_div = [](double numerator, double denominator) -> double {
        if (denominator == 0.0 ||
            !std::isfinite(numerator) ||
            !std::isfinite(denominator))
        {
            return 0.0;
        }
        const double value = numerator / denominator;
        return std::isfinite(value) ? value : 0.0;
    };
    const double acceptance_rate = safe_div(
        static_cast<double>(result.accepted_draft_token_count),
        static_cast<double>(result.drafted_token_count));
    const double mean_accepted = safe_div(
        static_cast<double>(result.accepted_draft_token_count),
        static_cast<double>(result.block_count));
    const double tokens_per_second = safe_div(
        static_cast<double>(decode.generated_tokens.size()),
        result.timing.total_ms / 1000.0);

    std::cout << label << ":\n";
    std::cout << "  seq_len: " << decode.seq_len << "\n";
    std::cout << "  checked_tokens: " << decode.checked_tokens << "\n";
    std::cout << "  generated_token_count: " << decode.generated_tokens.size() << "\n";
    std::cout << "  generated_text_chars: " << generated_text.size() << "\n";
    std::cout << "  block_count: " << result.block_count << "\n";
    std::cout << "  drafted_token_count: " << result.drafted_token_count << "\n";
    std::cout << "  accepted_draft_token_count: "
              << result.accepted_draft_token_count << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  timing_prefill_ms: " << result.timing.prefill_ms << "\n";
    std::cout << "  timing_draft_prepare_ms: " << result.timing.draft_prepare_ms << "\n";
    std::cout << "  timing_draft_infer_ms: " << result.timing.draft_infer_ms << "\n";
    std::cout << "  timing_draft_postprocess_ms: "
              << result.timing.draft_postprocess_ms << "\n";
    std::cout << "  timing_verify_prepare_ms: " << result.timing.verify_prepare_ms << "\n";
    std::cout << "  timing_verify_infer_ms: " << result.timing.verify_infer_ms << "\n";
    std::cout << "  timing_verify_postprocess_ms: "
              << result.timing.verify_postprocess_ms << "\n";
    std::cout << "  timing_commit_ms: " << result.timing.commit_ms << "\n";
    std::cout << "  timing_total_ms: " << result.timing.total_ms << "\n";
    std::cout << "  timing_tokens_per_second: " << tokens_per_second << "\n";
    std::cout << "  acceptance_rate: " << acceptance_rate << "\n";
    std::cout << "  mean_accepted_draft_tokens_per_block: " << mean_accepted << "\n";
    std::cout << "  target_verify_token_count: "
              << result.block_count * hunyuan_ocr::detail::kDFlashBlockSize << "\n";
    print_token_vector("  acceptance_lengths: ", result.acceptance_lengths);
    print_token_vector("  generated_tokens: ", decode.generated_tokens);
    if (!decode.expected_tokens.empty())
    {
        print_token_vector("  expected_tokens: ", decode.expected_tokens);
        std::cout << "  match_expected_tokens: "
                  << (decode.matches_expected() ? "true" : "false") << "\n";
    }

    if (!vlm_fixture_dir.empty())
    {
        std::string expected_text;
        if (read_text_file(vlm_fixture_dir + "/expected_text.txt", expected_text))
        {
            const bool text_match =
                strip_trailing_newlines(generated_text) == strip_trailing_newlines(expected_text);
            std::cout << "  match_expected_text: " << (text_match ? "true" : "false") << "\n";
            if (!text_match)
            {
                return 4;
            }
        }
    }
    std::cout << "Decoded text:\n" << generated_text << "\n";

    if (!decode.expected_tokens.empty() && !decode.matches_expected())
    {
        return 3;
    }
    return 0;
}

int run_vlm_decode_with_features(const std::string& label,
                                 const std::string& model_root,
                                 const std::string& vlm_fixture_dir,
                                 const std::vector<float>& vision_features,
                                 int vision_token_count,
                                 int max_tokens,
                                 int num_threads,
                                 bool use_dflash,
                                 bool mmap_weights)
{
    std::string error;
    hunyuan_ocr::TextRuntime text_runtime(num_threads, mmap_weights);
    if (!text_runtime.load(model_root, &error))
    {
        std::cerr << "Failed to load text runtime: " << error << "\n";
        return 1;
    }

    if (use_dflash)
    {
        if (!text_runtime.load_dflash(model_root, &error))
        {
            std::cerr << "Failed to load DFlash runtime: " << error << "\n";
            return 1;
        }
        hunyuan_ocr::DFlashDecodeResult decode;
        if (!text_runtime.run_vlm_fixture_dflash_decode_with_features(vlm_fixture_dir,
                                                                      vision_features,
                                                                      vision_token_count,
                                                                      max_tokens,
                                                                      &decode,
                                                                      &error))
        {
            std::cerr << label << " decode failed: " << error << "\n";
            return 1;
        }
        return print_dflash_decode_result(label, model_root, vlm_fixture_dir, decode);
    }

    hunyuan_ocr::TextDecodeResult decode;
    if (!text_runtime.run_vlm_fixture_decode_with_features(vlm_fixture_dir,
                                                           vision_features,
                                                           vision_token_count,
                                                           max_tokens,
                                                           &decode,
                                                           &error))
    {
        std::cerr << label << " decode failed: " << error << "\n";
        return 1;
    }

    hunyuan_ocr::Tokenizer tokenizer;
    if (!tokenizer.load(model_root + "/tokenizer/vocab.txt",
                        model_root + "/tokenizer/special_tokens.json",
                        &error))
    {
        std::cerr << "Failed to load tokenizer: " << error << "\n";
        return 1;
    }
    const std::string generated_text = tokenizer.decode(decode.generated_tokens, true);

    std::cout << label << ":\n";
    std::cout << "  seq_len: " << decode.seq_len << "\n";
    std::cout << "  checked_tokens: " << decode.checked_tokens << "\n";
    std::cout << "  generated_token_count: " << decode.generated_tokens.size() << "\n";
    std::cout << "  generated_text_chars: " << generated_text.size() << "\n";
    print_token_vector("  generated_tokens: ", decode.generated_tokens);
    print_token_vector("  expected_tokens: ", decode.expected_tokens);
    std::cout << "  match_expected_tokens: " << (decode.matches_expected() ? "true" : "false") << "\n";

    std::string expected_text;
    if (read_text_file(vlm_fixture_dir + "/expected_text.txt", expected_text))
    {
        const bool text_match = strip_trailing_newlines(generated_text) == strip_trailing_newlines(expected_text);
        std::cout << "  match_expected_text: " << (text_match ? "true" : "false") << "\n";
        if (!text_match)
        {
            return 4;
        }
    }
    std::cout << "Decoded text:\n" << generated_text << "\n";

    if (!decode.matches_expected())
    {
        return 3;
    }
    return 0;
}

int print_prompt_decode_result(const std::string& label,
                               const std::string& model_root,
                               const std::string& vlm_fixture_dir,
                               const hunyuan_ocr::TextDecodeResult& decode)
{
    std::string error;
    hunyuan_ocr::Tokenizer tokenizer;
    if (!tokenizer.load(model_root + "/tokenizer/vocab.txt",
                        model_root + "/tokenizer/special_tokens.json",
                        &error))
    {
        std::cerr << "Failed to load tokenizer: " << error << "\n";
        return 1;
    }
    const std::string generated_text = tokenizer.decode(decode.generated_tokens, true);

    std::cout << label << ":\n";
    std::cout << "  seq_len: " << decode.seq_len << "\n";
    std::cout << "  checked_tokens: " << decode.checked_tokens << "\n";
    std::cout << "  generated_token_count: " << decode.generated_tokens.size() << "\n";
    std::cout << "  generated_text_chars: " << generated_text.size() << "\n";
    print_token_vector("  generated_tokens: ", decode.generated_tokens);
    if (!decode.expected_tokens.empty())
    {
        print_token_vector("  expected_tokens: ", decode.expected_tokens);
        std::cout << "  match_expected_tokens: " << (decode.matches_expected() ? "true" : "false") << "\n";
    }

    if (!vlm_fixture_dir.empty())
    {
        std::string expected_text;
        if (read_text_file(vlm_fixture_dir + "/expected_text.txt", expected_text))
        {
            const bool text_match = strip_trailing_newlines(generated_text) == strip_trailing_newlines(expected_text);
            std::cout << "  match_expected_text: " << (text_match ? "true" : "false") << "\n";
            if (!text_match)
            {
                return 4;
            }
        }
    }
    std::cout << "Decoded text:\n" << generated_text << "\n";

    if (!decode.expected_tokens.empty() && !decode.matches_expected())
    {
        return 3;
    }
    return 0;
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

    hunyuan_ocr::TextRuntime text_runtime(num_threads, vision_options.mmap_weights);
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

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);

    int wide_argc = 0;
    LPWSTR* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv == nullptr)
    {
        std::cerr << "Failed to read the Windows command line.\n";
        return 1;
    }

    std::vector<std::wstring> wide_arguments;
    wide_arguments.reserve(static_cast<size_t>(wide_argc));
    for (int i = 0; i < wide_argc; ++i)
    {
        wide_arguments.emplace_back(wide_argv[i]);
    }
    LocalFree(wide_argv);

    std::vector<std::string> utf8_arguments;
    std::string utf8_error;
    if (!hunyuan_ocr::wide_arguments_to_utf8(wide_arguments, &utf8_arguments, &utf8_error))
    {
        std::cerr << "Failed to convert the Windows command line to UTF-8: " << utf8_error << "\n";
        return 1;
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_arguments.size());
    for (std::string& argument : utf8_arguments)
    {
        utf8_argv.push_back(argument.data());
    }
    argc = static_cast<int>(utf8_argv.size());
    argv = utf8_argv.data();
#endif
    const auto process_start = Clock::now();
    std::string model_root;
    std::string vlm_fixture_dir;
    bool dflash = false;
    bool mmap_weights = false;
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
    bool vision_vulkan_device_explicit = false;
    int vision_vulkan_device = 0;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--version")
        {
            std::cout << "HunyuanOCR-ncnn " << hunyuan_ocr::project_version() << "\n";
            std::cout << "ncnn " << hunyuan_ocr::ncnn_version() << "\n";
            std::cout << "vision Vulkan support: "
                      << (hunyuan_ocr::vision_vulkan_compiled() ? "enabled" : "disabled")
                      << "\n";
            return 0;
        }
        if (arg == "--model")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--model requires a path\n";
                return 1;
            }
            model_root = argv[++i];
            continue;
        }
        if (arg == "--vlm-fixture")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--vlm-fixture requires a path\n";
                return 1;
            }
            vlm_fixture_dir = argv[++i];
            continue;
        }
        if (arg == "--dflash")
        {
            dflash = true;
            continue;
        }
        if (arg == "--mmap-weights")
        {
            mmap_weights = true;
            continue;
        }
        if (arg == "--vision-vulkan")
        {
            vision_vulkan = true;
            continue;
        }
        if (arg == "--vision-vulkan-device")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--vision-vulkan-device requires an integer\n";
                return 1;
            }
            try
            {
                vision_vulkan_device = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--vision-vulkan-device value must be an integer\n";
                return 1;
            }
            vision_vulkan_device_explicit = true;
            continue;
        }
        if (arg == "--image")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--image requires a path\n";
                return 1;
            }
            image_path = argv[++i];
            continue;
        }
        if (arg == "--batch-input")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--batch-input requires a path\n";
                return 1;
            }
            batch_input_path = argv[++i];
            continue;
        }
        if (arg == "--batch-output")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--batch-output requires a path\n";
                return 1;
            }
            batch_output_path = argv[++i];
            continue;
        }
        if (arg == "--force")
        {
            force_batch_output = true;
            continue;
        }
        if (arg == "--prompt-mode")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--prompt-mode requires spotting or document\n";
                return 1;
            }
            prompt_mode_text = argv[++i];
            continue;
        }
        if (arg == "--prompt")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--prompt requires text\n";
                return 1;
            }
            prompt_text = argv[++i];
            continue;
        }
        if (arg == "--benchmark")
        {
            benchmark = true;
            continue;
        }
        if (arg == "--benchmark-warmup")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--benchmark-warmup requires an integer\n";
                return 1;
            }
            try
            {
                benchmark_warmup = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--benchmark-warmup value must be an integer\n";
                return 1;
            }
            continue;
        }
        if (arg == "--benchmark-repeat")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--benchmark-repeat requires an integer\n";
                return 1;
            }
            try
            {
                benchmark_repeat = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--benchmark-repeat value must be an integer\n";
                return 1;
            }
            continue;
        }
        if (arg == "--num-threads")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--num-threads requires an integer\n";
                return 1;
            }
            try
            {
                num_threads = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--num-threads value must be an integer\n";
                return 1;
            }
            continue;
        }
        if (arg == "--repetition-penalty")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--repetition-penalty requires a float\n";
                return 1;
            }
            try
            {
                repetition_penalty = std::stof(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--repetition-penalty value must be a float\n";
                return 1;
            }
            if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f)
            {
                std::cerr << "--repetition-penalty must be positive\n";
                return 1;
            }
            continue;
        }
        if (arg == "--max-tokens")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--max-tokens requires an integer\n";
                return 1;
            }
            try
            {
                max_tokens = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--max-tokens value must be an integer\n";
                return 1;
            }
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 1;
    }

    if (model_root.empty())
    {
        print_usage(argv[0]);
        return 0;
    }

    if (!prompt_text.empty() && !prompt_mode_text.empty())
    {
        std::cerr << "--prompt and --prompt-mode are mutually exclusive\n";
        return 1;
    }
    const bool has_batch_input = !batch_input_path.empty();
    const bool has_batch_output = !batch_output_path.empty();
    if (has_batch_input != has_batch_output)
    {
        std::cerr << "--batch-input and --batch-output must be provided together\n";
        return 1;
    }
    const bool batch_mode = has_batch_input && has_batch_output;
    if (force_batch_output && !batch_mode)
    {
        std::cerr << "--force requires batch mode\n";
        return 1;
    }
    if (batch_mode && !image_path.empty())
    {
        std::cerr << "--batch-input and --image are mutually exclusive\n";
        return 1;
    }
    if (batch_mode && (!prompt_text.empty() || !prompt_mode_text.empty()))
    {
        std::cerr << "batch prompts must be specified in JSONL records\n";
        return 1;
    }
    const bool batch_has_diagnostic_options = !vlm_fixture_dir.empty() || benchmark;
    if (batch_mode && batch_has_diagnostic_options)
    {
        std::cerr << "batch mode does not support diagnostic or benchmark options\n";
        return 1;
    }
    if (num_threads < 0)
    {
        std::cerr << "--num-threads must be positive when provided\n";
        return 1;
    }
    if (vision_vulkan_device_explicit && !vision_vulkan)
    {
        std::cerr << "--vision-vulkan-device requires --vision-vulkan\n";
        return 1;
    }
    if (vision_vulkan_device < 0)
    {
        std::cerr << "--vision-vulkan-device must be non-negative\n";
        return 1;
    }
    const bool image_executes_vision =
        !image_path.empty() &&
        (benchmark || !prompt_text.empty() || !prompt_mode_text.empty() ||
         !vlm_fixture_dir.empty());
    const bool executes_vision = batch_mode || image_executes_vision;
    if (vision_vulkan && !executes_vision)
    {
        std::cerr << "--vision-vulkan requires a path that executes vision\n";
        return 1;
    }
    if (vision_vulkan && !hunyuan_ocr::vision_vulkan_compiled())
    {
        std::cerr << "ncnn was built without Vulkan support\n";
        return 1;
    }
    const hunyuan_ocr::VisionRuntimeOptions vision_options =
        make_vision_runtime_options(
            num_threads, vision_vulkan, vision_vulkan_device, mmap_weights);
    if (benchmark_warmup < 0 || benchmark_repeat <= 0)
    {
        std::cerr << "benchmark warmup must be non-negative and repeat must be positive\n";
        return 1;
    }
    if (benchmark)
    {
        if (image_path.empty() || (prompt_mode_text.empty() && prompt_text.empty()))
        {
            std::cerr << "--benchmark requires --image and either --prompt-mode or --prompt\n";
            return 1;
        }
        if (max_tokens <= 0)
        {
            max_tokens = 64;
        }
    }
    if (benchmark && dflash)
    {
        std::cerr << "--benchmark does not support --dflash yet\n";
        return 1;
    }
    const bool has_image_prompt =
        !image_path.empty() && (!prompt_text.empty() || !prompt_mode_text.empty());
    if (dflash && vlm_fixture_dir.empty() && !has_image_prompt && !batch_mode)
    {
        std::cerr << "--dflash requires --vlm-fixture or --image with --prompt/--prompt-mode\n";
        return 1;
    }
    const hunyuan_ocr::ModelLayoutReport report = hunyuan_ocr::check_model_layout(model_root);
    const bool ready = report.required_files_present();

    if (!benchmark)
    {
        std::cout << "Model root: " << report.root << "\n";
        print_file_group("Present files:", report.present);
        print_file_group("Missing required files:", report.missing_required);
        print_file_group("Missing planned files:", report.missing_planned);
    }

    if (!ready)
    {
        std::cerr << "Model layout check failed: required files are missing.\n";
        return 2;
    }

    if (batch_mode)
    {
        hunyuan_ocr::detail::BatchOptions batch_options;
        batch_options.input_path = batch_input_path;
        batch_options.output_path = batch_output_path;
        batch_options.force = force_batch_output;
        batch_options.default_max_tokens = max_tokens > 0 ? max_tokens : 128;
        hunyuan_ocr::RuntimeError runtime_error;
        if (!hunyuan_ocr::detail::validate_jsonl_batch_io(batch_options, &runtime_error))
        {
            std::cerr << "Batch failed at " << runtime_error.stage
                      << ": " << runtime_error.message << "\n";
            return 1;
        }

        hunyuan_ocr::RuntimeOptions runtime_options;
        runtime_options.num_threads = num_threads;
        runtime_options.vision_vulkan = vision_vulkan;
        runtime_options.vision_vulkan_device = vision_vulkan_device;
        runtime_options.dflash = dflash;
        runtime_options.mmap_weights = mmap_weights;
        runtime_options.repetition_penalty = repetition_penalty;

        hunyuan_ocr::HunyuanOCR runtime;
        if (!runtime.load(model_root, runtime_options, &runtime_error))
        {
            std::cerr << "Runtime load failed at " << runtime_error.stage
                      << ": " << runtime_error.message << "\n";
            return 1;
        }

        hunyuan_ocr::detail::BatchSummary summary;
        const bool batch_ok = hunyuan_ocr::detail::run_jsonl_batch(
            batch_options,
            [&](const hunyuan_ocr::detail::BatchRequest& request,
                hunyuan_ocr::InferenceResult* result,
                hunyuan_ocr::RuntimeError* error) {
                return runtime.infer_file(request.resolved_image,
                                          request.request,
                                          result,
                                          error);
            },
            &summary,
            &runtime_error);
        if (!runtime_error.stage.empty())
        {
            std::cerr << "Batch failed at " << runtime_error.stage
                      << ": " << runtime_error.message << "\n";
            return 1;
        }
        std::cout << "Batch summary: " << summary.succeeded << "/" << summary.total
                  << " succeeded, " << summary.failed << " failed\n";
        return batch_ok ? 0 : 3;
    }

    if (benchmark)
    {
        return run_image_benchmark(model_root,
                                   image_path,
                                   prompt_mode_text,
                                   prompt_text,
                                   max_tokens,
                                   repetition_penalty,
                                   num_threads,
                                   benchmark_warmup,
                                   benchmark_repeat,
                                   vision_options,
                                   process_start);
    }

    std::cout << "Model layout check passed for the current runtime files.\n";

    const bool plain_image_inference = has_image_prompt && vlm_fixture_dir.empty();
    if (plain_image_inference)
    {
        hunyuan_ocr::RuntimeOptions runtime_options;
        runtime_options.num_threads = num_threads;
        runtime_options.vision_vulkan = vision_vulkan;
        runtime_options.vision_vulkan_device = vision_vulkan_device;
        runtime_options.dflash = dflash;
        runtime_options.mmap_weights = mmap_weights;
        runtime_options.repetition_penalty = repetition_penalty;

        hunyuan_ocr::RuntimeError runtime_error;
        hunyuan_ocr::HunyuanOCR runtime;
        if (!runtime.load(model_root, runtime_options, &runtime_error))
        {
            std::cerr << "Runtime load failed at " << runtime_error.stage
                      << ": " << runtime_error.message << "\n";
            return 1;
        }

        hunyuan_ocr::InferenceRequest request;
        request.max_tokens = max_tokens > 0 ? max_tokens : 128;
        if (!prompt_text.empty())
        {
            request.prompt_mode = hunyuan_ocr::PromptMode::Custom;
            request.prompt = prompt_text;
        }
        else
        {
            std::string prompt_error;
            if (!hunyuan_ocr::parse_prompt_mode(prompt_mode_text,
                                                &request.prompt_mode,
                                                &prompt_error))
            {
                std::cerr << "Prompt mode failed: " << prompt_error << "\n";
                return 1;
            }
        }

        hunyuan_ocr::InferenceResult result;
        if (!runtime.infer_file(image_path, request, &result, &runtime_error))
        {
            std::cerr << "Inference failed at " << runtime_error.stage
                      << ": " << runtime_error.message << "\n";
            return 1;
        }

        std::cout << "Image:\n";
        std::cout << "  path: " << image_path << "\n";
        std::cout << "Inference:\n";
        std::cout << "  decoder: "
                  << (result.decoder.mode == hunyuan_ocr::DecoderMode::DFlash ? "dflash" : "ar")
                  << "\n";
        std::cout << "  generated_token_count: " << result.token_ids.size() << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  timing_preprocess_ms: " << result.timing.preprocess_ms << "\n";
        std::cout << "  timing_vision_ms: " << result.timing.vision_ms << "\n";
        std::cout << "  timing_text_ms: " << result.timing.text_ms << "\n";
        std::cout << "  timing_total_ms: " << result.timing.total_ms << "\n";
        if (result.decoder.mode == hunyuan_ocr::DecoderMode::DFlash)
        {
            std::cout << "  dflash_blocks: " << result.decoder.block_count << "\n";
            std::cout << "  dflash_drafted_tokens: "
                      << result.decoder.drafted_token_count << "\n";
            std::cout << "  dflash_accepted_tokens: "
                      << result.decoder.accepted_draft_token_count << "\n";
        }
        print_token_vector("  generated_tokens: ", result.token_ids);
        std::cout << "Decoded text:\n" << result.text << "\n";
        return 0;
    }

    const bool pure_dflash_fixture = dflash && image_path.empty();
    if (pure_dflash_fixture)
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads, mmap_weights);
        if (!text_runtime.load(model_root, &error) ||
            !text_runtime.load_dflash(model_root, &error))
        {
            std::cerr << "Failed to load DFlash runtime: " << error << "\n";
            return 1;
        }

        hunyuan_ocr::DFlashDecodeResult result;
        if (!text_runtime.run_vlm_fixture_dflash_decode(vlm_fixture_dir,
                                                        max_tokens,
                                                        &result,
                                                        &error))
        {
            std::cerr << "DFlash fixture decode failed: " << error << "\n";
            return 1;
        }
        return print_dflash_decode_result("DFlash fixture decode",
                                          model_root,
                                          vlm_fixture_dir,
                                          result);
    }

    if (!image_path.empty())
    {
        std::string error;
        hunyuan_ocr::ImagePreprocessor preprocessor;
        hunyuan_ocr::ImagePreprocessResult image;
        if (!preprocessor.preprocess_image_file(image_path, &image, &error))
        {
            std::cerr << "Image preprocess failed: " << error << "\n";
            return 1;
        }

        std::cout << "Image:\n";
        std::cout << "  path: " << image_path << "\n";
        std::cout << "  original_size: " << image.original_width << "x" << image.original_height << "\n";
        std::cout << "  resized_size: " << image.resized_width << "x" << image.resized_height << "\n";
        std::cout << "  image_grid_thw: [" << image.grid_t << ", " << image.grid_h << ", " << image.grid_w << "]\n";
        std::cout << "  patch_count: " << image.patch_count << "\n";
        std::cout << "  pixel_value_count: " << image.pixel_value_count << "\n";

        std::string resolved_vision_param_path;
        std::string resolved_vision_bin_path;
        std::string resolved_pos_embed_path;
        const bool wants_image_decode =
            !prompt_mode_text.empty() || !prompt_text.empty() || !vlm_fixture_dir.empty();
        if (wants_image_decode &&
            !resolve_dynamic_vision_paths(model_root,
                                          resolved_vision_param_path,
                                          resolved_vision_bin_path,
                                          resolved_pos_embed_path))
        {
            std::cerr << "Canonical dynamic vision files are missing under "
                      << model_root << "/vision\n";
            return 1;
        }

        if (wants_image_decode)
        {
            hunyuan_ocr::VisionRuntime vision_runtime(vision_options);
            if (!vision_runtime.load_dynamic(resolved_vision_param_path,
                                             resolved_vision_bin_path,
                                             resolved_pos_embed_path,
                                             &error))
            {
                std::cerr << "Failed to load dynamic vision runtime: " << error << "\n";
                return 1;
            }
            print_vision_compute_backend(vision_options, vision_runtime);

            hunyuan_ocr::VisionRuntimeResult vision;
            if (!vision_runtime.run_dynamic_pixel_values(image.pixel_values,
                                                         image.grid_h,
                                                         image.grid_w,
                                                         preprocessor.config().merge_size,
                                                         &vision,
                                                         &error))
            {
                std::cerr << "Image + vision failed: " << error << "\n";
                return 1;
            }

            std::cout << "Image + vision:\n";
            std::cout << "  backend: dynamic\n";
            std::cout << "  param: " << resolved_vision_param_path << "\n";
            std::cout << "  bin: " << resolved_vision_bin_path << "\n";
            std::cout << "  pos_embed: " << resolved_pos_embed_path << "\n";
            std::cout << "  patch_count: " << vision.patch_count << "\n";
            std::cout << "  vision_token_count: " << vision.vision_token_count << "\n";
            std::cout << "  feature_values: " << vision.feature_values << "\n";

            if (!print_fixture_vision_diff(vision.vision_features, vlm_fixture_dir, error))
            {
                std::cerr << "Vision fixture comparison failed: " << error << "\n";
                return 1;
            }

            if (!prompt_mode_text.empty() || !prompt_text.empty())
            {
                hunyuan_ocr::PromptBuildResult prompt;
                std::string prompt_label;
                if (!prompt_text.empty())
                {
                    hunyuan_ocr::Tokenizer tokenizer;
                    if (!tokenizer.load(model_root + "/tokenizer/vocab.txt",
                                        model_root + "/tokenizer/merges.txt",
                                        model_root + "/tokenizer/special_tokens.json",
                                        &error))
                    {
                        std::cerr << "Failed to load tokenizer for prompt encode: " << error << "\n";
                        return 1;
                    }
                    const std::vector<int> prompt_token_ids = tokenizer.encode(prompt_text, &error);
                    if (!error.empty())
                    {
                        std::cerr << "Prompt encode failed: " << error << "\n";
                        return 1;
                    }
                    if (!hunyuan_ocr::build_hunyuan_ocr_prompt_from_tokens(prompt_token_ids,
                                                                           image.grid_h,
                                                                           image.grid_w,
                                                                           preprocessor.config().merge_size,
                                                                           &prompt,
                                                                           &error))
                    {
                        std::cerr << "Prompt build failed: " << error << "\n";
                        return 1;
                    }
                    prompt_label = "custom";
                }
                else
                {
                    hunyuan_ocr::PromptMode prompt_mode = hunyuan_ocr::PromptMode::Spotting;
                    if (!hunyuan_ocr::parse_prompt_mode(prompt_mode_text, &prompt_mode, &error))
                    {
                        std::cerr << "Prompt mode failed: " << error << "\n";
                        return 1;
                    }
                    if (!hunyuan_ocr::build_hunyuan_ocr_prompt(prompt_mode,
                                                               image.grid_h,
                                                               image.grid_w,
                                                               preprocessor.config().merge_size,
                                                               &prompt,
                                                               &error))
                    {
                        std::cerr << "Prompt build failed: " << error << "\n";
                        return 1;
                    }
                    prompt_label = hunyuan_ocr::prompt_mode_name(prompt_mode);
                }

                std::cout << "Prompt:\n";
                std::cout << "  mode: " << prompt_label << "\n";
                std::cout << "  chat_template_token_count: " << prompt.chat_template_ids.size() << "\n";
                std::cout << "  seq_len: " << prompt.seq_len << "\n";
                std::cout << "  vision_token_count: " << prompt.vision_token_count << "\n";

                std::vector<int> expected_tokens;
                if (!vlm_fixture_dir.empty())
                {
                    std::vector<int> fixture_input_ids;
                    std::vector<int> fixture_position_ids;
                    if (!read_binary_vector_file(vlm_fixture_dir + "/input_ids.i32", fixture_input_ids, error) ||
                        !read_binary_vector_file(vlm_fixture_dir + "/position_ids.i32", fixture_position_ids, error) ||
                        !read_binary_vector_file(vlm_fixture_dir + "/expected_tokens.i32", expected_tokens, error))
                    {
                        std::cerr << "Failed to read prompt comparison fixture: " << error << "\n";
                        return 1;
                    }
                    const bool input_ids_match = fixture_input_ids == prompt.input_ids;
                    const bool position_ids_match = fixture_position_ids == prompt.position_ids;
                    std::cout << "  match_fixture_input_ids: " << (input_ids_match ? "true" : "false") << "\n";
                    std::cout << "  match_fixture_position_ids: " << (position_ids_match ? "true" : "false") << "\n";
                    if (!input_ids_match || !position_ids_match)
                    {
                        return 7;
                    }
                }

                hunyuan_ocr::TextRuntime text_runtime(num_threads, mmap_weights);
                if (!text_runtime.load(model_root, &error))
                {
                    std::cerr << "Failed to load text runtime: " << error << "\n";
                    return 1;
                }

                if (dflash)
                {
                    if (!text_runtime.load_dflash(model_root, &error))
                    {
                        std::cerr << "Failed to load DFlash runtime: " << error << "\n";
                        return 1;
                    }

                    hunyuan_ocr::DFlashDecodeResult decode;
                    if (!text_runtime.run_vlm_dflash_decode_with_prompt(prompt.input_ids,
                                                                        prompt.position_ids,
                                                                        prompt.image_token_id,
                                                                        vision.vision_features,
                                                                        vision.vision_token_count,
                                                                        expected_tokens,
                                                                        max_tokens,
                                                                        repetition_penalty,
                                                                        &decode,
                                                                        &error))
                    {
                        std::cerr << "Image + prompt + vision DFlash decode failed: "
                                  << error << "\n";
                        return 1;
                    }

                    const int rc = print_dflash_decode_result(
                        "Image + prompt + vision DFlash decode",
                        model_root,
                        vlm_fixture_dir,
                        decode);
                    if (rc != 0)
                    {
                        return rc;
                    }
                }
                else
                {
                    hunyuan_ocr::TextDecodeResult decode;
                    if (!text_runtime.run_vlm_decode_with_prompt(prompt.input_ids,
                                                                 prompt.position_ids,
                                                                 prompt.image_token_id,
                                                                 vision.vision_features,
                                                                 vision.vision_token_count,
                                                                 expected_tokens,
                                                                 max_tokens,
                                                                 repetition_penalty,
                                                                 &decode,
                                                                 &error))
                    {
                        std::cerr << "Image + prompt + vision decode failed: " << error << "\n";
                        return 1;
                    }

                    const int rc = print_prompt_decode_result("Image + prompt + vision decode",
                                                              model_root,
                                                              vlm_fixture_dir,
                                                              decode);
                    if (rc != 0)
                    {
                        return rc;
                    }
                }
            }
            else if (!vlm_fixture_dir.empty())
            {
                const int rc = run_vlm_decode_with_features("Image + vision + VLM fixture decode",
                                                            model_root,
                                                            vlm_fixture_dir,
                                                            vision.vision_features,
                                                            vision.vision_token_count,
                                                            max_tokens,
                                                            num_threads,
                                                            dflash,
                                                            mmap_weights);
                if (rc != 0)
                {
                    return rc;
                }
            }
        }
    }

    if (!dflash &&
        !vlm_fixture_dir.empty() &&
        image_path.empty() &&
        prompt_mode_text.empty() &&
        prompt_text.empty())
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads, mmap_weights);
        if (!text_runtime.load(model_root, &error))
        {
            std::cerr << "Failed to load text runtime: " << error << "\n";
            return 1;
        }

        hunyuan_ocr::TextDecodeResult decode;
        if (!text_runtime.run_vlm_fixture_decode(vlm_fixture_dir, max_tokens, &decode, &error))
        {
            std::cerr << "VLM fixture decode failed: " << error << "\n";
            return 1;
        }

        hunyuan_ocr::Tokenizer tokenizer;
        if (!tokenizer.load(model_root + "/tokenizer/vocab.txt",
                            model_root + "/tokenizer/special_tokens.json",
                            &error))
        {
            std::cerr << "Failed to load tokenizer: " << error << "\n";
            return 1;
        }
        const std::string generated_text = tokenizer.decode(decode.generated_tokens, true);

        const auto safe_div = [](double num, double den) -> double {
            if (den == 0.0 || !std::isfinite(den) || !std::isfinite(num))
            {
                return 0.0;
            }
            const double value = num / den;
            return std::isfinite(value) ? value : 0.0;
        };
        const double tokens_per_second = safe_div(
            static_cast<double>(decode.generated_tokens.size()),
            decode.timing.total_ms / 1000.0);

        std::cout << "VLM fixture decode:\n";
        std::cout << "  seq_len: " << decode.seq_len << "\n";
        std::cout << "  checked_tokens: " << decode.checked_tokens << "\n";
        std::cout << "  generated_token_count: " << decode.generated_tokens.size() << "\n";
        std::cout << "  generated_text_chars: " << generated_text.size() << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  timing_prefill_ms: " << decode.timing.prefill_ms << "\n";
        std::cout << "  timing_text_embed_ms: " << decode.timing.text_embed_ms << "\n";
        std::cout << "  timing_decode_ms: " << decode.timing.decode_ms << "\n";
        std::cout << "  timing_lm_head_ms: " << decode.timing.lm_head_ms << "\n";
        std::cout << "  timing_token_select_ms: " << decode.timing.token_select_ms << "\n";
        std::cout << "  timing_total_ms: " << decode.timing.total_ms << "\n";
        std::cout << "  timing_tokens_per_second: " << tokens_per_second << "\n";
        print_token_vector("  generated_tokens: ", decode.generated_tokens);
        print_token_vector("  expected_tokens: ", decode.expected_tokens);
        std::cout << "  match_expected_tokens: " << (decode.matches_expected() ? "true" : "false") << "\n";

        std::string expected_text;
        if (read_text_file(vlm_fixture_dir + "/expected_text.txt", expected_text))
        {
            const bool text_match = strip_trailing_newlines(generated_text) == strip_trailing_newlines(expected_text);
            std::cout << "  match_expected_text: " << (text_match ? "true" : "false") << "\n";
            if (!text_match)
            {
                return 4;
            }
        }
        std::cout << "Decoded text:\n" << generated_text << "\n";

        if (!decode.matches_expected())
        {
            return 3;
        }
    }

    return 0;
}
