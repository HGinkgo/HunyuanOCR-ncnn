#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/image_preprocessor.h"
#include "hunyuan_ocr/prompt_builder.h"
#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/tokenizer.h"
#include "hunyuan_ocr/vision_runtime.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [--help] [--version] [--model PATH]\n"
        << "\n"
        << "HunyuanOCR-ncnn CLI. It validates model layout, runs development\n"
        << "fixtures, and executes the fixed-grid PNG/JPEG image path.\n"
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
        << "  --vision-param PATH    Load a per-shape vision ncnn param file for fixture validation.\n"
        << "  --vision-bin PATH      Load a per-shape vision ncnn bin file for fixture validation.\n"
        << "  --vision-fixture PATH  Run pixel_values -> vision_features fixture.\n"
        << "  --vision-tolerance F   Max absolute diff tolerance for expected_vision_features.f32.\n"
        << "  --image PATH           Run PNG/JPEG image -> resize -> flattened pixel_values.\n"
        << "  --prompt-mode MODE     Build built-in prompt tensors in C++: spotting or document.\n"
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
    std::ifstream file(path);
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
    const std::filesystem::path vision_dir = std::filesystem::path(model_root) / "vision" / grid_name;
    const std::filesystem::path candidate_param = vision_dir / "vision.ncnn.param";
    const std::filesystem::path candidate_bin = vision_dir / "vision.ncnn.bin";
    if (!file_exists(candidate_param) || !file_exists(candidate_bin))
    {
        return false;
    }

    param_path = candidate_param.string();
    bin_path = candidate_bin.string();
    return true;
}

template <typename T>
bool read_binary_vector_file(const std::string& path, std::vector<T>& out, std::string& error)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
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

int run_vlm_decode_with_features(const std::string& label,
                                 const std::string& model_root,
                                 const std::string& vlm_fixture_dir,
                                 const std::vector<float>& vision_features,
                                 int vision_token_count,
                                 int max_tokens)
{
    std::string error;
    hunyuan_ocr::TextRuntime text_runtime;
    if (!text_runtime.load(model_root, &error))
    {
        std::cerr << "Failed to load text runtime: " << error << "\n";
        return 1;
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

} // namespace

int main(int argc, char** argv)
{
    std::string model_root;
    std::string decode_ids_text;
    std::string decode_ids_file;
    bool skip_special_tokens = true;
    bool run_text_smoke = false;
    int smoke_token_id = 0;
    std::string text_fixture_dir;
    std::string vlm_fixture_dir;
    std::string vision_param_path;
    std::string vision_bin_path;
    std::string vision_fixture_dir;
    float vision_tolerance = 0.002f;
    std::string image_path;
    std::string prompt_mode_text;
    std::string image_preprocess_fixture_dir;
    std::string image_file_fixture_dir;
    float image_preprocess_tolerance = 0.000001f;
    int max_tokens = 0;

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
            std::cout << "HunyuanOCR-ncnn 0.1.0\n";
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

    hunyuan_ocr::HunyuanOCR runtime;
    const bool ready = runtime.load(model_root);
    const auto& report = runtime.layout_report();

    std::cout << "Model root: " << report.root << "\n";
    print_file_group("Present files:", report.present);
    print_file_group("Missing required files:", report.missing_required);
    print_file_group("Missing planned files:", report.missing_planned);

    if (!ready)
    {
        std::cerr << "Model layout check failed: required files are missing.\n";
        return 2;
    }

    std::cout << "Model layout check passed for the current text-runtime scaffold.\n";

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
        hunyuan_ocr::TextRuntime text_runtime;
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
        hunyuan_ocr::TextRuntime text_runtime;
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
        const bool wants_image_decode =
            !prompt_mode_text.empty() || !vlm_fixture_dir.empty() ||
            !resolved_vision_param_path.empty() || !resolved_vision_bin_path.empty();
        if (wants_image_decode &&
            !resolve_fixed_grid_vision_paths(model_root,
                                             image.grid_h,
                                             image.grid_w,
                                             resolved_vision_param_path,
                                             resolved_vision_bin_path))
        {
            if (vision_param_path.empty() && vision_bin_path.empty())
            {
                std::cerr << "No fixed-grid vision artifact found for grid "
                          << image.grid_h << "x" << image.grid_w
                          << " under " << model_root << "/vision/grid_"
                          << image.grid_h << "x" << image.grid_w
                          << ". Pass --vision-param and --vision-bin or package that grid.\n";
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

            hunyuan_ocr::VisionRuntime vision_runtime;
            if (!vision_runtime.load(resolved_vision_param_path, resolved_vision_bin_path, &error))
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
                std::cerr << "Image + vision failed: " << error << "\n";
                return 1;
            }

            std::cout << "Image + vision:\n";
            std::cout << "  param: " << resolved_vision_param_path << "\n";
            std::cout << "  bin: " << resolved_vision_bin_path << "\n";
            std::cout << "  patch_count: " << vision.patch_count << "\n";
            std::cout << "  vision_token_count: " << vision.vision_token_count << "\n";
            std::cout << "  feature_values: " << vision.feature_values << "\n";

            if (!prompt_mode_text.empty())
            {
                hunyuan_ocr::PromptMode prompt_mode = hunyuan_ocr::PromptMode::Spotting;
                if (!hunyuan_ocr::parse_prompt_mode(prompt_mode_text, &prompt_mode, &error))
                {
                    std::cerr << "Prompt mode failed: " << error << "\n";
                    return 1;
                }

                hunyuan_ocr::PromptBuildResult prompt;
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

                std::cout << "Prompt:\n";
                std::cout << "  mode: " << hunyuan_ocr::prompt_mode_name(prompt_mode) << "\n";
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

                hunyuan_ocr::TextRuntime text_runtime;
                if (!text_runtime.load(model_root, &error))
                {
                    std::cerr << "Failed to load text runtime: " << error << "\n";
                    return 1;
                }

                hunyuan_ocr::TextDecodeResult decode;
                if (!text_runtime.run_vlm_decode_with_prompt(prompt.input_ids,
                                                             prompt.position_ids,
                                                             prompt.image_token_id,
                                                             vision.vision_features,
                                                             vision.vision_token_count,
                                                             expected_tokens,
                                                             max_tokens,
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
            else if (!vlm_fixture_dir.empty())
            {
                const int rc = run_vlm_decode_with_features("Image + vision + VLM fixture decode",
                                                            model_root,
                                                            vlm_fixture_dir,
                                                            vision.vision_features,
                                                            vision.vision_token_count,
                                                            max_tokens);
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

            hunyuan_ocr::VisionRuntime vision_runtime;
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
                                                            max_tokens);
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

            hunyuan_ocr::VisionRuntime vision_runtime;
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
                                                            max_tokens);
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
        hunyuan_ocr::VisionRuntime vision_runtime;
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
            hunyuan_ocr::TextRuntime text_runtime;
            if (!text_runtime.load(model_root, &error))
            {
                std::cerr << "Failed to load text runtime: " << error << "\n";
                return 1;
            }

            hunyuan_ocr::TextDecodeResult decode;
            if (!text_runtime.run_vlm_fixture_decode_with_features(vlm_fixture_dir,
                                                                   vision.vision_features,
                                                                   vision.vision_token_count,
                                                                   max_tokens,
                                                                   &decode,
                                                                   &error))
            {
                std::cerr << "Vision+VLM fixture decode failed: " << error << "\n";
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

            std::cout << "Vision+VLM fixture decode:\n";
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
        }

        if (vision.has_expected_features && !vision.matches_expected(vision_tolerance))
        {
            return 5;
        }
    }

    if (!vlm_fixture_dir.empty() &&
        vision_fixture_dir.empty() &&
        image_path.empty() &&
        prompt_mode_text.empty() &&
        image_preprocess_fixture_dir.empty() &&
        image_file_fixture_dir.empty())
    {
        std::string error;
        hunyuan_ocr::TextRuntime text_runtime;
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

        std::cout << "VLM fixture decode:\n";
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
    }

    return 0;
}
