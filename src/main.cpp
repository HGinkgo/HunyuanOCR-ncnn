#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/image_preprocessor.h"
#include "hunyuan_ocr/prompt_builder.h"
#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/tokenizer.h"
#include "hunyuan_ocr/utf8.h"
#include "hunyuan_ocr/vision_runtime.h"

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

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [--help] [--version] [--model PATH]\n"
        << "\n"
        << "HunyuanOCR-ncnn CLI. It validates model layout, runs development\n"
        << "fixtures, and executes the PNG/JPEG image path.\n"
        << "\n"
        << "Options:\n"
        << "  --help          Show this help message.\n"
        << "  --version       Print project and ncnn version information.\n"
        << "  --model PATH    Check a HunyuanOCR ncnn model directory.\n"
        << "  --decode-ids IDS       Decode comma/space separated token ids with tokenizer files under --model.\n"
        << "  --decode-ids-file PATH Decode token ids read from a text file.\n"
        << "  --no-skip-special      Keep special tokens when decoding token ids.\n"
        << "  --smoke-text TOKEN_ID  Load text ncnn nets and run text_embed + lm_head smoke.\n"
        << "  --text-fixture PATH    Run decoder prefill/KV greedy decode from a raw tensor fixture.\n"
        << "  --vlm-fixture PATH     Run input_ids + vision_features + text decode fixture.\n"
        << "  --dflash               Run DFlash with --vlm-fixture or --image with --prompt/--prompt-mode.\n"
        << "  --dflash-probe         Run one 16-token DFlash block with --vlm-fixture.\n"
        << "  --vision-param PATH    Load a per-grid vision ncnn param file for fixture validation.\n"
        << "  --vision-bin PATH      Load a per-grid vision ncnn bin file for fixture validation.\n"
        << "  --vision-fixture PATH  Run pixel_values -> vision_features fixture.\n"
        << "  --vision-tolerance F   Max absolute diff tolerance for expected_vision_features.f32.\n"
        << "  --image PATH           Run PNG/JPEG image -> resize -> flattened pixel_values.\n"
        << "  --prompt-mode MODE     Build built-in prompt tensors in C++: spotting or document.\n"
        << "  --prompt TEXT          Build a custom image prompt with the bundled tokenizer.\n"
        << "  --benchmark            Run cold-start and same-process warm inference timing.\n"
        << "  --benchmark-warmup N   Same-process warmup iterations. Default: 0.\n"
        << "  --benchmark-repeat N   Same-process measured iterations. Default: 1.\n"
        << "  --num-threads N        ncnn thread count for all submodels.\n"
        << "  --repetition-penalty F Greedy decode repetition penalty. Default: 1.08.\n"
        << "  --image-preprocess-fixture PATH Run resized RGB -> flattened pixel_values fixture.\n"
        << "  --image-preprocess-tolerance F  Max absolute diff tolerance for expected_pixel_values.f32.\n"
        << "  --image-file-fixture PATH Run original RGB -> resize -> flattened pixel_values fixture.\n"
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

bool resolve_fixed_grid_vision_paths(const std::string& model_root,
                                     int grid_h,
                                     int grid_w,
                                     std::string& param_path,
                                     std::string& bin_path)
{
    if (!param_path.empty() || !bin_path.empty())
    {
        return !param_path.empty() && !bin_path.empty();
    }

    const std::string grid_name = "grid_" + std::to_string(grid_h) + "x" + std::to_string(grid_w);
    const std::filesystem::path vision_dir = hunyuan_ocr::path_from_utf8(model_root) / "vision" / grid_name;
    const std::filesystem::path candidate_param = vision_dir / "vision.ncnn.param";
    const std::filesystem::path candidate_bin = vision_dir / "vision.ncnn.bin";
    if (!file_exists(candidate_param) || !file_exists(candidate_bin))
    {
        return false;
    }

    param_path = hunyuan_ocr::path_to_utf8(candidate_param);
    bin_path = hunyuan_ocr::path_to_utf8(candidate_bin);
    return true;
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

    const std::filesystem::path legacy_param = vision_dir / "vision_dynamic.ncnn.param";
    const std::filesystem::path legacy_bin = vision_dir / "vision_dynamic.ncnn.bin";
    if (file_exists(legacy_param) && file_exists(legacy_bin) && file_exists(pos_path))
    {
        param_path = hunyuan_ocr::path_to_utf8(legacy_param);
        bin_path = hunyuan_ocr::path_to_utf8(legacy_bin);
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
                                 bool use_dflash)
{
    std::string error;
    hunyuan_ocr::TextRuntime text_runtime(num_threads);
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
                             bool use_dynamic_vision,
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

    const int patch_h = image->grid_h / preprocessor.config().merge_size;
    const int patch_w = image->grid_w / preprocessor.config().merge_size;
    const int vision_token_count = patch_h * (patch_w + 1) + 2;
    hunyuan_ocr::VisionRuntimeResult vision;
    const auto vision_start = Clock::now();
    const bool vision_ok = use_dynamic_vision
        ? vision_runtime.run_dynamic_pixel_values(image->pixel_values,
                                                  image->grid_h,
                                                  image->grid_w,
                                                  preprocessor.config().merge_size,
                                                  &vision,
                                                  error)
        : vision_runtime.run_pixel_values(image->pixel_values,
                                          image->patch_count,
                                          vision_token_count,
                                          &vision,
                                          error);
    if (!vision_ok)
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
                        const std::string& vision_param_path,
                        const std::string& vision_bin_path,
                        int max_tokens,
                        float repetition_penalty,
                        int num_threads,
                        int warmup,
                        int repeat,
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

    std::string resolved_vision_param_path = vision_param_path;
    std::string resolved_vision_bin_path = vision_bin_path;
    std::string resolved_pos_embed_path;
    bool use_dynamic_vision = false;
    const bool explicit_vision_paths = !vision_param_path.empty() || !vision_bin_path.empty();
    if (!explicit_vision_paths)
    {
        use_dynamic_vision = resolve_dynamic_vision_paths(model_root,
                                                          resolved_vision_param_path,
                                                          resolved_vision_bin_path,
                                                          resolved_pos_embed_path);
    }
    if (!use_dynamic_vision &&
        !resolve_fixed_grid_vision_paths(model_root,
                                         probe_image.grid_h,
                                         probe_image.grid_w,
                                         resolved_vision_param_path,
                                         resolved_vision_bin_path))
    {
        std::cerr << "Benchmark could not resolve a dynamic or fixed-grid vision model\n";
        return 1;
    }

    hunyuan_ocr::VisionRuntime vision_runtime(num_threads);
    const auto vision_load_start = Clock::now();
    const bool vision_loaded = use_dynamic_vision
        ? vision_runtime.load_dynamic(resolved_vision_param_path,
                                      resolved_vision_bin_path,
                                      resolved_pos_embed_path,
                                      &error)
        : vision_runtime.load(resolved_vision_param_path, resolved_vision_bin_path, &error);
    if (!vision_loaded)
    {
        std::cerr << "Benchmark vision load failed: " << error << "\n";
        return 1;
    }
    const double vision_load_ms = elapsed_ms(vision_load_start, Clock::now());

    hunyuan_ocr::TextRuntime text_runtime(num_threads);
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

    BenchmarkIteration cold;
    if (!run_benchmark_iteration(image_path,
                                 prompt_mode_text,
                                 prompt_text,
                                 max_tokens,
                                 repetition_penalty,
                                 use_dynamic_vision,
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
                                     use_dynamic_vision,
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
                                     use_dynamic_vision,
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
    std::string decode_ids_text;
    std::string decode_ids_file;
    bool skip_special_tokens = true;
    bool run_text_smoke = false;
    int smoke_token_id = 0;
    std::string text_fixture_dir;
    std::string vlm_fixture_dir;
    bool dflash = false;
    bool dflash_probe = false;
    std::string vision_param_path;
    std::string vision_bin_path;
    std::string vision_fixture_dir;
    float vision_tolerance = 0.002f;
    std::string image_path;
    std::string prompt_mode_text;
    std::string prompt_text;
    std::string image_preprocess_fixture_dir;
    std::string image_file_fixture_dir;
    float image_preprocess_tolerance = 0.000001f;
    int max_tokens = 0;
    float repetition_penalty = 1.08f;
    bool benchmark = false;
    int benchmark_warmup = 0;
    int benchmark_repeat = 1;
    int num_threads = 0;

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
        if (arg == "--decode-ids")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--decode-ids requires a token id list\n";
                return 1;
            }
            decode_ids_text = argv[++i];
            continue;
        }
        if (arg == "--decode-ids-file")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--decode-ids-file requires a path\n";
                return 1;
            }
            decode_ids_file = argv[++i];
            continue;
        }
        if (arg == "--no-skip-special")
        {
            skip_special_tokens = false;
            continue;
        }
        if (arg == "--smoke-text")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--smoke-text requires a token id\n";
                return 1;
            }
            try
            {
                smoke_token_id = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--smoke-text token id must be an integer\n";
                return 1;
            }
            run_text_smoke = true;
            continue;
        }
        if (arg == "--text-fixture")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--text-fixture requires a path\n";
                return 1;
            }
            text_fixture_dir = argv[++i];
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
        if (arg == "--dflash-probe")
        {
            dflash_probe = true;
            continue;
        }
        if (arg == "--dflash")
        {
            dflash = true;
            continue;
        }
        if (arg == "--vision-param")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--vision-param requires a path\n";
                return 1;
            }
            vision_param_path = argv[++i];
            continue;
        }
        if (arg == "--vision-bin")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--vision-bin requires a path\n";
                return 1;
            }
            vision_bin_path = argv[++i];
            continue;
        }
        if (arg == "--vision-fixture")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--vision-fixture requires a path\n";
                return 1;
            }
            vision_fixture_dir = argv[++i];
            continue;
        }
        if (arg == "--vision-tolerance")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--vision-tolerance requires a float\n";
                return 1;
            }
            try
            {
                vision_tolerance = std::stof(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--vision-tolerance value must be a float\n";
                return 1;
            }
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
        if (arg == "--image-preprocess-fixture")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--image-preprocess-fixture requires a path\n";
                return 1;
            }
            image_preprocess_fixture_dir = argv[++i];
            continue;
        }
        if (arg == "--image-file-fixture")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--image-file-fixture requires a path\n";
                return 1;
            }
            image_file_fixture_dir = argv[++i];
            continue;
        }
        if (arg == "--image-preprocess-tolerance")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--image-preprocess-tolerance requires a float\n";
                return 1;
            }
            try
            {
                image_preprocess_tolerance = std::stof(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--image-preprocess-tolerance value must be a float\n";
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

    if (!decode_ids_file.empty())
    {
        if (!read_text_file(decode_ids_file, decode_ids_text))
        {
            std::cerr << "Failed to read token id file: " << decode_ids_file << "\n";
            return 1;
        }
    }
    if (!prompt_text.empty() && !prompt_mode_text.empty())
    {
        std::cerr << "--prompt and --prompt-mode are mutually exclusive\n";
        return 1;
    }
    if (num_threads < 0)
    {
        std::cerr << "--num-threads must be positive when provided\n";
        return 1;
    }
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
    if (dflash && dflash_probe)
    {
        std::cerr << "--dflash and --dflash-probe are mutually exclusive\n";
        return 1;
    }
    if (benchmark && dflash)
    {
        std::cerr << "--benchmark does not support --dflash yet\n";
        return 1;
    }
    const bool has_image_prompt =
        !image_path.empty() && (!prompt_text.empty() || !prompt_mode_text.empty());
    if (dflash && vlm_fixture_dir.empty() && !has_image_prompt)
    {
        std::cerr << "--dflash requires --vlm-fixture or --image with --prompt/--prompt-mode\n";
        return 1;
    }
    if (dflash_probe && vlm_fixture_dir.empty())
    {
        std::cerr << "--dflash-probe requires --vlm-fixture\n";
        return 1;
    }

    hunyuan_ocr::HunyuanOCR runtime;
    const bool ready = runtime.load(model_root);
    const auto& report = runtime.layout_report();

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

    if (benchmark)
    {
        return run_image_benchmark(model_root,
                                   image_path,
                                   prompt_mode_text,
                                   prompt_text,
                                   vision_param_path,
                                   vision_bin_path,
                                   max_tokens,
                                   repetition_penalty,
                                   num_threads,
                                   benchmark_warmup,
                                   benchmark_repeat,
                                   process_start);
    }

    std::cout << "Model layout check passed for the current runtime files.\n";

    if (dflash_probe)
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads);
        if (!text_runtime.load(model_root, &error) ||
            !text_runtime.load_dflash(model_root, &error))
        {
            std::cerr << "Failed to load DFlash probe runtime: " << error << "\n";
            return 1;
        }

        hunyuan_ocr::DFlashBlockProbeResult probe;
        if (!text_runtime.run_vlm_fixture_dflash_probe(vlm_fixture_dir, &probe, &error))
        {
            std::cerr << "DFlash block probe failed: " << error << "\n";
            return 1;
        }
        std::cout << "DFlash block probe:\n";
        std::cout << "  seq_len: " << probe.seq_len << "\n";
        std::cout << "  first_token: " << probe.first_token << "\n";
        std::cout << "  acceptance_length: " << probe.acceptance_length << "\n";
        std::cout << "  correction_token: " << probe.correction_token << "\n";
        print_token_vector("  proposed_tokens: ", probe.proposed_tokens);
        print_token_vector("  target_tokens: ", probe.target_tokens);
        std::cout << "  match_expected_first_token: "
                  << (probe.first_token_matches_expected ? "true" : "false") << "\n";
        std::cout << "  match_expected_first_target_token: "
                  << (probe.first_target_token_matches_expected ? "true" : "false") << "\n";
        return probe.first_token_matches_expected &&
                       probe.first_target_token_matches_expected
                   ? 0
                   : 3;
    }

    const bool pure_dflash_fixture = dflash &&
        image_path.empty() &&
        image_file_fixture_dir.empty() &&
        image_preprocess_fixture_dir.empty() &&
        vision_fixture_dir.empty();
    if (pure_dflash_fixture)
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads);
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

    if (!decode_ids_text.empty())
    {
        std::string error;
        const std::vector<int> ids = hunyuan_ocr::parse_token_ids(decode_ids_text, &error);
        if (!error.empty())
        {
            std::cerr << "Failed to parse token ids: " << error << "\n";
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
        std::cout << "Decoded text:\n" << tokenizer.decode(ids, skip_special_tokens) << "\n";
    }

    if (run_text_smoke)
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads);
        if (!text_runtime.load(model_root, &error))
        {
            std::cerr << "Failed to load text runtime: " << error << "\n";
            return 1;
        }
        const hunyuan_ocr::TextRuntimeSmokeResult smoke = text_runtime.smoke_token(smoke_token_id, &error);
        if (!error.empty())
        {
            std::cerr << "Text runtime smoke failed: " << error << "\n";
            return 1;
        }
        std::cout << "Text runtime smoke:\n";
        std::cout << "  token_id: " << smoke.token_id << "\n";
        std::cout << "  embedding_values: " << smoke.embedding_values << "\n";
        std::cout << "  embedding_shape: w=" << smoke.embedding_w
                  << " h=" << smoke.embedding_h
                  << " c=" << smoke.embedding_c
                  << " elempack=" << smoke.embedding_elempack << "\n";
        std::cout << "  logits_values: " << smoke.logits_values << "\n";
        std::cout << "  logits_shape: w=" << smoke.logits_w
                  << " h=" << smoke.logits_h
                  << " c=" << smoke.logits_c
                  << " elempack=" << smoke.logits_elempack << "\n";
        std::cout << "  raw_top1: " << smoke.raw_top1 << "\n";
        std::cout << "  raw_top1_score: " << smoke.raw_top1_score << "\n";
    }

    if (!text_fixture_dir.empty())
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads);
        if (!text_runtime.load(model_root, &error))
        {
            std::cerr << "Failed to load text runtime: " << error << "\n";
            return 1;
        }

        hunyuan_ocr::TextDecodeResult decode;
        if (!text_runtime.run_fixture_decode(text_fixture_dir, max_tokens, &decode, &error))
        {
            std::cerr << "Text fixture decode failed: " << error << "\n";
            return 1;
        }

        std::cout << "Text fixture decode:\n";
        std::cout << "  seq_len: " << decode.seq_len << "\n";
        std::cout << "  checked_tokens: " << decode.checked_tokens << "\n";
        std::cout << "  repetition_penalty: " << decode.repetition_penalty << "\n";
        print_token_vector("  generated_tokens: ", decode.generated_tokens);
        print_token_vector("  raw_top1_tokens: ", decode.raw_top1_tokens);
        print_token_vector("  expected_tokens: ", decode.expected_tokens);
        std::cout << "  match_expected: " << (decode.matches_expected() ? "true" : "false") << "\n";
        if (!decode.matches_expected())
        {
            return 3;
        }
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

        std::string resolved_vision_param_path = vision_param_path;
        std::string resolved_vision_bin_path = vision_bin_path;
        std::string resolved_pos_embed_path;
        bool use_dynamic_vision = false;
        const bool explicit_vision_paths = !vision_param_path.empty() || !vision_bin_path.empty();
        const bool wants_image_decode =
            !prompt_mode_text.empty() || !prompt_text.empty() || !vlm_fixture_dir.empty() ||
            !resolved_vision_param_path.empty() || !resolved_vision_bin_path.empty();
        if (wants_image_decode && !explicit_vision_paths)
        {
            use_dynamic_vision = resolve_dynamic_vision_paths(model_root,
                                                              resolved_vision_param_path,
                                                              resolved_vision_bin_path,
                                                              resolved_pos_embed_path);
        }
        if (wants_image_decode &&
            !use_dynamic_vision &&
            !resolve_fixed_grid_vision_paths(model_root,
                                             image.grid_h,
                                             image.grid_w,
                                             resolved_vision_param_path,
                                             resolved_vision_bin_path))
        {
            if (vision_param_path.empty() && vision_bin_path.empty())
            {
                std::cerr << "No dynamic or fixed-grid vision artifact found for image_grid_thw=["
                          << image.grid_t << "," << image.grid_h << "," << image.grid_w << "]"
                          << " under " << model_root << "/vision. Expected either vision/vision.ncnn.param,"
                          << " vision/vision.ncnn.bin, vision/pos_embed.bin, or vision/grid_"
                          << image.grid_h << "x" << image.grid_w
                          << "/vision.ncnn.param/bin. Pass --vision-param and --vision-bin or package a supported vision backend.\n";
            }
            else
            {
                std::cerr << "--image with vision requires both --vision-param and --vision-bin\n";
            }
            return 1;
        }

        if (!resolved_vision_param_path.empty() || !resolved_vision_bin_path.empty())
        {
            if (resolved_vision_param_path.empty() || resolved_vision_bin_path.empty())
            {
                std::cerr << "--image with vision requires both --vision-param and --vision-bin\n";
                return 1;
            }

            hunyuan_ocr::VisionRuntime vision_runtime(num_threads);
            if (use_dynamic_vision)
            {
                if (!vision_runtime.load_dynamic(resolved_vision_param_path,
                                                 resolved_vision_bin_path,
                                                 resolved_pos_embed_path,
                                                 &error))
                {
                    std::cerr << "Failed to load dynamic vision runtime: " << error << "\n";
                    return 1;
                }
            }
            else if (!vision_runtime.load(resolved_vision_param_path, resolved_vision_bin_path, &error))
            {
                std::cerr << "Failed to load vision runtime: " << error << "\n";
                return 1;
            }

            const int patch_h = image.grid_h / preprocessor.config().merge_size;
            const int patch_w = image.grid_w / preprocessor.config().merge_size;
            const int vision_token_count = patch_h * (patch_w + 1) + 2;

            hunyuan_ocr::VisionRuntimeResult vision;
            const bool vision_ok = use_dynamic_vision
                ? vision_runtime.run_dynamic_pixel_values(image.pixel_values,
                                                          image.grid_h,
                                                          image.grid_w,
                                                          preprocessor.config().merge_size,
                                                          &vision,
                                                          &error)
                : vision_runtime.run_pixel_values(image.pixel_values,
                                                  image.patch_count,
                                                  vision_token_count,
                                                  &vision,
                                                  &error);
            if (!vision_ok)
            {
                std::cerr << "Image + vision failed: " << error << "\n";
                return 1;
            }

            std::cout << "Image + vision:\n";
            std::cout << "  backend: " << (use_dynamic_vision ? "dynamic" : "fixed-grid") << "\n";
            std::cout << "  param: " << resolved_vision_param_path << "\n";
            std::cout << "  bin: " << resolved_vision_bin_path << "\n";
            if (use_dynamic_vision)
            {
                std::cout << "  pos_embed: " << resolved_pos_embed_path << "\n";
            }
            std::cout << "  patch_count: " << vision.patch_count << "\n";
            std::cout << "  vision_token_count: " << vision.vision_token_count << "\n";
            std::cout << "  feature_values: " << vision.feature_values << "\n";

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

                hunyuan_ocr::TextRuntime text_runtime(num_threads);
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
                                                            dflash);
                if (rc != 0)
                {
                    return rc;
                }
            }
        }
    }

    if (!image_file_fixture_dir.empty())
    {
        std::string error;
        hunyuan_ocr::ImagePreprocessor preprocessor;
        hunyuan_ocr::ImagePreprocessResult image;
        if (!preprocessor.run_image_file_fixture(image_file_fixture_dir, &image, &error))
        {
            std::cerr << "Image file fixture failed: " << error << "\n";
            return 1;
        }

        std::cout << "Image file fixture:\n";
        std::cout << "  original_size: " << image.original_width << "x" << image.original_height << "\n";
        std::cout << "  resized_size: " << image.resized_width << "x" << image.resized_height << "\n";
        std::cout << "  image_grid_thw: [" << image.grid_t << ", " << image.grid_h << ", " << image.grid_w << "]\n";
        std::cout << "  patch_count: " << image.patch_count << "\n";
        std::cout << "  pixel_value_count: " << image.pixel_value_count << "\n";
        if (image.has_expected_pixel_values)
        {
            std::cout << "  max_abs_diff_expected: " << image.max_abs_diff_expected << "\n";
            std::cout << "  mean_abs_diff_expected: " << image.mean_abs_diff_expected << "\n";
            std::cout << "  match_expected_pixel_values: "
                      << (image.matches_expected(image_preprocess_tolerance) ? "true" : "false") << "\n";
            if (!image.matches_expected(image_preprocess_tolerance))
            {
                return 6;
            }
        }

        if (!vision_param_path.empty() || !vision_bin_path.empty())
        {
            if (vision_param_path.empty() || vision_bin_path.empty())
            {
                std::cerr << "--image-file-fixture with vision requires both --vision-param and --vision-bin\n";
                return 1;
            }

            hunyuan_ocr::VisionRuntime vision_runtime(num_threads);
            if (!vision_runtime.load(vision_param_path, vision_bin_path, &error))
            {
                std::cerr << "Failed to load vision runtime: " << error << "\n";
                return 1;
            }

            const int patch_h = image.grid_h / preprocessor.config().merge_size;
            const int patch_w = image.grid_w / preprocessor.config().merge_size;
            const int vision_token_count = patch_h * (patch_w + 1) + 2;

            hunyuan_ocr::VisionRuntimeResult vision;
            if (!vision_runtime.run_pixel_values(image.pixel_values,
                                                 image.patch_count,
                                                 vision_token_count,
                                                 &vision,
                                                 &error))
            {
                std::cerr << "Image file + vision failed: " << error << "\n";
                return 1;
            }

            std::cout << "Image file + vision:\n";
            std::cout << "  patch_count: " << vision.patch_count << "\n";
            std::cout << "  vision_token_count: " << vision.vision_token_count << "\n";
            std::cout << "  feature_values: " << vision.feature_values << "\n";

            if (!vlm_fixture_dir.empty())
            {
                const int rc = run_vlm_decode_with_features("Image file + vision + VLM fixture decode",
                                                            model_root,
                                                            vlm_fixture_dir,
                                                            vision.vision_features,
                                                            vision.vision_token_count,
                                                            max_tokens,
                                                            num_threads,
                                                            dflash);
                if (rc != 0)
                {
                    return rc;
                }
            }
        }
    }

    if (!image_preprocess_fixture_dir.empty())
    {
        std::string error;
        hunyuan_ocr::ImagePreprocessor preprocessor;
        hunyuan_ocr::ImagePreprocessResult image;
        if (!preprocessor.run_fixture(image_preprocess_fixture_dir, &image, &error))
        {
            std::cerr << "Image preprocess fixture failed: " << error << "\n";
            return 1;
        }

        std::cout << "Image preprocess fixture:\n";
        std::cout << "  original_size: " << image.original_width << "x" << image.original_height << "\n";
        std::cout << "  resized_size: " << image.resized_width << "x" << image.resized_height << "\n";
        std::cout << "  image_grid_thw: [" << image.grid_t << ", " << image.grid_h << ", " << image.grid_w << "]\n";
        std::cout << "  patch_count: " << image.patch_count << "\n";
        std::cout << "  pixel_value_count: " << image.pixel_value_count << "\n";
        if (image.has_expected_pixel_values)
        {
            std::cout << "  max_abs_diff_expected: " << image.max_abs_diff_expected << "\n";
            std::cout << "  mean_abs_diff_expected: " << image.mean_abs_diff_expected << "\n";
            std::cout << "  match_expected_pixel_values: "
                      << (image.matches_expected(image_preprocess_tolerance) ? "true" : "false") << "\n";
            if (!image.matches_expected(image_preprocess_tolerance))
            {
                return 6;
            }
        }

        if (!vision_param_path.empty() || !vision_bin_path.empty())
        {
            if (vision_param_path.empty() || vision_bin_path.empty())
            {
                std::cerr << "--image-preprocess-fixture with vision requires both --vision-param and --vision-bin\n";
                return 1;
            }

            hunyuan_ocr::VisionRuntime vision_runtime(num_threads);
            if (!vision_runtime.load(vision_param_path, vision_bin_path, &error))
            {
                std::cerr << "Failed to load vision runtime: " << error << "\n";
                return 1;
            }

            const int patch_h = image.grid_h / preprocessor.config().merge_size;
            const int patch_w = image.grid_w / preprocessor.config().merge_size;
            const int vision_token_count = patch_h * (patch_w + 1) + 2;

            hunyuan_ocr::VisionRuntimeResult vision;
            if (!vision_runtime.run_pixel_values(image.pixel_values,
                                                 image.patch_count,
                                                 vision_token_count,
                                                 &vision,
                                                 &error))
            {
                std::cerr << "Image preprocess + vision failed: " << error << "\n";
                return 1;
            }

            std::cout << "Image preprocess + vision:\n";
            std::cout << "  patch_count: " << vision.patch_count << "\n";
            std::cout << "  vision_token_count: " << vision.vision_token_count << "\n";
            std::cout << "  feature_values: " << vision.feature_values << "\n";

            if (!vlm_fixture_dir.empty())
            {
                const int rc = run_vlm_decode_with_features("Image preprocess + vision + VLM fixture decode",
                                                            model_root,
                                                            vlm_fixture_dir,
                                                            vision.vision_features,
                                                            vision.vision_token_count,
                                                            max_tokens,
                                                            num_threads,
                                                            dflash);
                if (rc != 0)
                {
                    return rc;
                }
            }
        }
    }

    if (!vision_fixture_dir.empty())
    {
        if (vision_param_path.empty() || vision_bin_path.empty())
        {
            std::cerr << "--vision-fixture requires --vision-param and --vision-bin\n";
            return 1;
        }

        std::string error;
        hunyuan_ocr::VisionRuntime vision_runtime(num_threads);
        if (!vision_runtime.load(vision_param_path, vision_bin_path, &error))
        {
            std::cerr << "Failed to load vision runtime: " << error << "\n";
            return 1;
        }

        hunyuan_ocr::VisionRuntimeResult vision;
        if (!vision_runtime.run_fixture(vision_fixture_dir, &vision, &error))
        {
            std::cerr << "Vision fixture failed: " << error << "\n";
            return 1;
        }

        std::cout << "Vision fixture:\n";
        std::cout << "  patch_count: " << vision.patch_count << "\n";
        std::cout << "  vision_token_count: " << vision.vision_token_count << "\n";
        std::cout << "  feature_values: " << vision.feature_values << "\n";
        if (vision.has_expected_features)
        {
            std::cout << "  max_abs_diff_expected: " << vision.max_abs_diff_expected << "\n";
            std::cout << "  mean_abs_diff_expected: " << vision.mean_abs_diff_expected << "\n";
            std::cout << "  match_expected_features: "
                      << (vision.matches_expected(vision_tolerance) ? "true" : "false") << "\n";
        }

        if (!vlm_fixture_dir.empty())
        {
            const int rc = run_vlm_decode_with_features("Vision+VLM fixture decode",
                                                        model_root,
                                                        vlm_fixture_dir,
                                                        vision.vision_features,
                                                        vision.vision_token_count,
                                                        max_tokens,
                                                        num_threads,
                                                        dflash);
            if (rc != 0)
            {
                return rc;
            }
        }

        if (vision.has_expected_features && !vision.matches_expected(vision_tolerance))
        {
            return 5;
        }
    }

    if (!dflash &&
        !vlm_fixture_dir.empty() &&
        vision_fixture_dir.empty() &&
        image_path.empty() &&
        prompt_mode_text.empty() &&
        image_preprocess_fixture_dir.empty() &&
        image_file_fixture_dir.empty())
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads);
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
