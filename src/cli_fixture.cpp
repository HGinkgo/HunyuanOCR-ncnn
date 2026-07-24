#include "cli_fixture.h"

#include "hunyuan_ocr/image_preprocessor.h"
#include "hunyuan_ocr/prompt_builder.h"
#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/tokenizer.h"
#include "hunyuan_ocr/utf8.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace hunyuan_ocr::cli {

void print_vision_compute_backend(const VisionRuntimeOptions& options,
                                  const VisionRuntime& runtime)
{
    if (!options.use_vulkan) return;
    std::cout << "vision_backend: vulkan\n";
    std::cout << "vision_vulkan_device: " << options.vulkan_device << "\n";
    std::cout << "vision_gelu_cpu_fallback_count: "
              << runtime.gelu_cpu_fallback_count() << "\n";
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
                                 bool mmap_weights,
                                 bool text_vulkan,
                                 int text_vulkan_device)
{
    std::string error;
    hunyuan_ocr::TextRuntime text_runtime(num_threads,
                                          mmap_weights,
                                          text_vulkan,
                                          text_vulkan_device);
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


int run_fixture_mode(const CliOptions& options)
{
    const std::string& model_root = options.model_root;
    const std::string& vlm_fixture_dir = options.vlm_fixture_dir;
    const bool dflash = options.dflash;
    const bool mmap_weights = options.mmap_weights;
    const std::string& image_path = options.image_path;
    const std::string& prompt_mode_text = options.prompt_mode_text;
    const std::string& prompt_text = options.prompt_text;
    const int max_tokens = options.max_tokens;
    const float repetition_penalty = options.repetition_penalty;
    const int num_threads = options.num_threads;
    const bool text_vulkan = options.text_vulkan;
    const int text_vulkan_device = options.text_vulkan_device;
    const VisionRuntimeOptions& vision_options = options.vision_options;

    const bool pure_dflash_fixture = dflash && image_path.empty();
    if (pure_dflash_fixture)
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime(num_threads,
                                              mmap_weights,
                                              text_vulkan,
                                              text_vulkan_device);
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

                hunyuan_ocr::TextRuntime text_runtime(num_threads,
                                                      mmap_weights,
                                                      text_vulkan,
                                                      text_vulkan_device);
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
                                                            mmap_weights,
                                                            text_vulkan,
                                                            text_vulkan_device);
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
        hunyuan_ocr::TextRuntime text_runtime(num_threads,
                                              mmap_weights,
                                              text_vulkan,
                                              text_vulkan_device);
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

} // namespace hunyuan_ocr::cli
