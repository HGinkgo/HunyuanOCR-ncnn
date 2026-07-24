#include "cli_options.h"

#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/utf8.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace hunyuan_ocr::cli {
namespace {

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [--help] [--version] [--model PATH]\n"
        << "\n"
        << "HunyuanOCR-ncnn CLI for PNG/JPEG and JSONL inference.\n"
        << "\n"
        << "Common options:\n"
        << "  --help          Show this help message.\n"
        << "  --version       Print project and ncnn version information.\n"
        << "  --model PATH    Use a model directory. Auto-detects common model directories.\n"
        << "  --image PATH           Run one-image OCR. Default prompt: document.\n"
        << "  --interactive          Keep the model loaded for repeated image OCR.\n"
        << "  --vulkan               Run vision and text with ncnn Vulkan fp32.\n"
        << "  --vulkan-device N      Vulkan device index. Default: 0.\n"
        << "  --prompt-mode MODE     Build built-in prompt tensors in C++: spotting or document.\n"
        << "  --prompt TEXT          Build a custom image prompt with the bundled tokenizer.\n"
        << "  --num-threads N        ncnn thread count for all submodels.\n"
        << "  --max-tokens N         Limit generated token count. Default max tokens: 8192.\n"
        << "\n"
        << "Batch options:\n"
        << "  --batch-input PATH     Read ordered inference requests from a JSONL file.\n"
        << "  --batch-output PATH    Write one ordered JSON result per input line.\n"
        << "  --force                Replace an existing --batch-output file.\n"
        << "\n"
        << "Advanced options:\n"
        << "  --dflash               Run DFlash with --vlm-fixture or a single --image.\n"
        << "  --mmap-weights         Load model weights from read-only file mappings.\n"
        << "  --repetition-penalty F Greedy decode repetition penalty. Default: 1.08.\n"
        << "  --benchmark            Run cold-start and same-process warm inference timing.\n"
        << "  --benchmark-warmup N   Same-process warmup iterations. Default: 0.\n"
        << "  --benchmark-repeat N   Same-process measured iterations. Default: 1.\n"
        << "  --vlm-fixture PATH     Run input_ids + vision_features + text decode fixture.\n";
}

void print_version()
{
    std::cout << "HunyuanOCR-ncnn " << project_version() << "\n";
    std::cout << "ncnn " << ncnn_version() << "\n";
    std::cout << "vision Vulkan support: "
              << (vision_vulkan_compiled() ? "enabled" : "disabled") << "\n";
    std::cout << "text Vulkan support: "
              << (text_vulkan_compiled() ? "enabled" : "disabled") << "\n";
}

void append_model_candidates(const std::filesystem::path& base,
                             std::vector<std::filesystem::path>* candidates)
{
    if (base.empty()) return;
    candidates->push_back(base / "hunyuan_ocr_ncnn_model");
    candidates->push_back(base / "assets" / "hunyuan_ocr_1_5");
    candidates->push_back(base / "assets" / "hunyuan_ocr");
}

std::string discover_model_root(const char* program)
{
    std::vector<std::filesystem::path> candidates;
    std::error_code filesystem_error;
    const std::filesystem::path current = std::filesystem::current_path(filesystem_error);
    if (!filesystem_error) append_model_candidates(current, &candidates);

    filesystem_error.clear();
    std::filesystem::path executable = path_from_utf8(program);
    executable = std::filesystem::absolute(executable, filesystem_error);
    if (!filesystem_error)
    {
        const std::filesystem::path executable_dir = executable.parent_path();
        append_model_candidates(executable_dir, &candidates);
        append_model_candidates(executable_dir.parent_path(), &candidates);
    }

    std::string first_existing;
    for (const std::filesystem::path& candidate : candidates)
    {
        filesystem_error.clear();
        if (!std::filesystem::is_directory(candidate, filesystem_error)) continue;

        filesystem_error.clear();
        const std::filesystem::path absolute =
            std::filesystem::absolute(candidate, filesystem_error);
        const std::string path = path_to_utf8(
            filesystem_error ? candidate : absolute.lexically_normal());
        if (first_existing.empty()) first_existing = path;
        if (check_model_layout(path).required_files_present()) return path;
    }
    return first_existing;
}

bool require_value(int argc, char** argv, int* index, const char* message, std::string* value)
{
    if (*index + 1 >= argc)
    {
        std::cerr << message << "\n";
        return false;
    }
    *value = argv[++*index];
    return true;
}

bool parse_integer(int argc,
                   char** argv,
                   int* index,
                   const char* missing_message,
                   const char* invalid_message,
                   int* value)
{
    if (*index + 1 >= argc)
    {
        std::cerr << missing_message << "\n";
        return false;
    }
    try
    {
        *value = std::stoi(argv[++*index]);
    }
    catch (const std::exception&)
    {
        std::cerr << invalid_message << "\n";
        return false;
    }
    return true;
}

bool parse_arguments(int argc, char** argv, CliOptions* options, bool* exit_success)
{
    bool vulkan_device_explicit = false;
    bool vision_vulkan_device_explicit = false;
    bool text_vulkan_device_explicit = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            *exit_success = true;
            return false;
        }
        if (arg == "--version")
        {
            print_version();
            *exit_success = true;
            return false;
        }
        if (arg == "--model")
        {
            if (!require_value(argc, argv, &i, "--model requires a path", &options->model_root))
                return false;
            continue;
        }
        if (arg == "--vlm-fixture")
        {
            if (!require_value(argc, argv, &i, "--vlm-fixture requires a path",
                               &options->vlm_fixture_dir))
                return false;
            continue;
        }
        if (arg == "--dflash") { options->dflash = true; continue; }
        if (arg == "--mmap-weights") { options->mmap_weights = true; continue; }
        if (arg == "--interactive") { options->interactive = true; continue; }
        if (arg == "--vulkan") { options->vulkan = true; continue; }
        if (arg == "--vulkan-device")
        {
            if (!parse_integer(argc, argv, &i, "--vulkan-device requires an integer",
                               "--vulkan-device value must be an integer",
                               &options->vulkan_device))
                return false;
            vulkan_device_explicit = true;
            continue;
        }
        if (arg == "--vision-vulkan") { options->vision_vulkan = true; continue; }
        if (arg == "--vision-vulkan-device")
        {
            if (!parse_integer(argc, argv, &i, "--vision-vulkan-device requires an integer",
                               "--vision-vulkan-device value must be an integer",
                               &options->vision_vulkan_device))
                return false;
            vision_vulkan_device_explicit = true;
            continue;
        }
        if (arg == "--text-vulkan") { options->text_vulkan = true; continue; }
        if (arg == "--text-vulkan-device")
        {
            if (!parse_integer(argc, argv, &i, "--text-vulkan-device requires an integer",
                               "--text-vulkan-device value must be an integer",
                               &options->text_vulkan_device))
                return false;
            text_vulkan_device_explicit = true;
            continue;
        }
        if (arg == "--image")
        {
            if (!require_value(argc, argv, &i, "--image requires a path", &options->image_path))
                return false;
            continue;
        }
        if (arg == "--batch-input")
        {
            if (!require_value(argc, argv, &i, "--batch-input requires a path",
                               &options->batch_input_path))
                return false;
            continue;
        }
        if (arg == "--batch-output")
        {
            if (!require_value(argc, argv, &i, "--batch-output requires a path",
                               &options->batch_output_path))
                return false;
            continue;
        }
        if (arg == "--force") { options->force_batch_output = true; continue; }
        if (arg == "--prompt-mode")
        {
            if (!require_value(argc, argv, &i, "--prompt-mode requires spotting or document",
                               &options->prompt_mode_text))
                return false;
            continue;
        }
        if (arg == "--prompt")
        {
            if (!require_value(argc, argv, &i, "--prompt requires text", &options->prompt_text))
                return false;
            continue;
        }
        if (arg == "--benchmark") { options->benchmark = true; continue; }
        if (arg == "--benchmark-warmup")
        {
            if (!parse_integer(argc, argv, &i, "--benchmark-warmup requires an integer",
                               "--benchmark-warmup value must be an integer",
                               &options->benchmark_warmup))
                return false;
            continue;
        }
        if (arg == "--benchmark-repeat")
        {
            if (!parse_integer(argc, argv, &i, "--benchmark-repeat requires an integer",
                               "--benchmark-repeat value must be an integer",
                               &options->benchmark_repeat))
                return false;
            continue;
        }
        if (arg == "--num-threads")
        {
            if (!parse_integer(argc, argv, &i, "--num-threads requires an integer",
                               "--num-threads value must be an integer", &options->num_threads))
                return false;
            continue;
        }
        if (arg == "--repetition-penalty")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--repetition-penalty requires a float\n";
                return false;
            }
            try
            {
                options->repetition_penalty = std::stof(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--repetition-penalty value must be a float\n";
                return false;
            }
            if (!std::isfinite(options->repetition_penalty) || options->repetition_penalty <= 0.0f)
            {
                std::cerr << "--repetition-penalty must be positive\n";
                return false;
            }
            continue;
        }
        if (arg == "--max-tokens")
        {
            if (!parse_integer(argc, argv, &i, "--max-tokens requires an integer",
                               "--max-tokens value must be an integer", &options->max_tokens))
                return false;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return false;
    }

    if (options->model_root.empty())
    {
        options->model_root = discover_model_root(argv[0]);
        if (options->model_root.empty())
        {
            std::cerr << "No model directory found. Pass --model PATH or place the model at "
                      << "./hunyuan_ocr_ncnn_model.\n";
            return false;
        }
        std::cerr << "Auto-detected model: " << options->model_root << "\n";
    }

    if (!options->prompt_text.empty() && !options->prompt_mode_text.empty())
    {
        std::cerr << "--prompt and --prompt-mode are mutually exclusive\n";
        return false;
    }
    const bool has_batch_input = !options->batch_input_path.empty();
    const bool has_batch_output = !options->batch_output_path.empty();
    if (has_batch_input != has_batch_output)
    {
        std::cerr << "--batch-input and --batch-output must be provided together\n";
        return false;
    }
    options->batch_mode = has_batch_input && has_batch_output;
    if (options->interactive && !options->image_path.empty())
    {
        std::cerr << "--interactive and --image are mutually exclusive\n";
        return false;
    }
    if (options->interactive && options->batch_mode)
    {
        std::cerr << "--interactive and batch mode are mutually exclusive\n";
        return false;
    }
    if (options->interactive && (!options->vlm_fixture_dir.empty() || options->benchmark))
    {
        std::cerr << "--interactive does not support diagnostic or benchmark options\n";
        return false;
    }
    if (options->force_batch_output && !options->batch_mode)
    {
        std::cerr << "--force requires batch mode\n";
        return false;
    }
    if (options->batch_mode && !options->image_path.empty())
    {
        std::cerr << "--batch-input and --image are mutually exclusive\n";
        return false;
    }
    if (options->batch_mode &&
        (!options->prompt_text.empty() || !options->prompt_mode_text.empty()))
    {
        std::cerr << "batch prompts must be specified in JSONL records\n";
        return false;
    }
    if (options->batch_mode && (!options->vlm_fixture_dir.empty() || options->benchmark))
    {
        std::cerr << "batch mode does not support diagnostic or benchmark options\n";
        return false;
    }
    if (options->num_threads < 0)
    {
        std::cerr << "--num-threads must be positive when provided\n";
        return false;
    }
    if (vulkan_device_explicit && !options->vulkan)
    {
        std::cerr << "--vulkan-device requires --vulkan\n";
        return false;
    }
    if (options->vulkan_device < 0)
    {
        std::cerr << "--vulkan-device must be non-negative\n";
        return false;
    }
    if (options->vulkan && options->dflash)
    {
        std::cerr << "--vulkan cannot be combined with --dflash yet\n";
        return false;
    }
    if (options->vulkan)
    {
        options->vision_vulkan = true;
        options->vision_vulkan_device = options->vulkan_device;
        options->text_vulkan = true;
        options->text_vulkan_device = options->vulkan_device;
    }
    if (vision_vulkan_device_explicit && !options->vision_vulkan)
    {
        std::cerr << "--vision-vulkan-device requires --vision-vulkan\n";
        return false;
    }
    if (options->vision_vulkan_device < 0)
    {
        std::cerr << "--vision-vulkan-device must be non-negative\n";
        return false;
    }
    if (text_vulkan_device_explicit && !options->text_vulkan)
    {
        std::cerr << "--text-vulkan-device requires --text-vulkan\n";
        return false;
    }
    if (options->text_vulkan_device < 0)
    {
        std::cerr << "--text-vulkan-device must be non-negative\n";
        return false;
    }
    if (options->text_vulkan && options->dflash)
    {
        std::cerr << "--text-vulkan cannot be combined with --dflash yet\n";
        return false;
    }
    options->plain_image_inference = !options->image_path.empty() &&
                                     options->vlm_fixture_dir.empty() &&
                                     !options->benchmark;
    const bool executes_vision = options->batch_mode || !options->image_path.empty() ||
                                 options->interactive;
    if (options->vision_vulkan && !executes_vision)
    {
        std::cerr << "--vision-vulkan requires a path that executes vision\n";
        return false;
    }
    if (options->vision_vulkan && !vision_vulkan_compiled())
    {
        std::cerr << "ncnn was built without Vulkan support\n";
        return false;
    }
    if (options->text_vulkan && !text_vulkan_compiled())
    {
        std::cerr << "ncnn was built without Vulkan support\n";
        return false;
    }
    options->vision_options.num_threads = options->num_threads;
    options->vision_options.use_vulkan = options->vision_vulkan;
    options->vision_options.vulkan_device = options->vision_vulkan_device;
    options->vision_options.mmap_weights = options->mmap_weights;
    if (options->benchmark_warmup < 0 || options->benchmark_repeat <= 0)
    {
        std::cerr << "benchmark warmup must be non-negative and repeat must be positive\n";
        return false;
    }
    if (options->benchmark)
    {
        if (options->image_path.empty() ||
            (options->prompt_mode_text.empty() && options->prompt_text.empty()))
        {
            std::cerr << "--benchmark requires --image and either --prompt-mode or --prompt\n";
            return false;
        }
        if (options->max_tokens <= 0) options->max_tokens = 64;
    }
    if (options->benchmark && options->dflash)
    {
        std::cerr << "--benchmark does not support --dflash yet\n";
        return false;
    }
    const bool has_image_prompt = options->plain_image_inference ||
        (!options->image_path.empty() &&
         (!options->prompt_text.empty() || !options->prompt_mode_text.empty()));
    if (options->dflash && options->vlm_fixture_dir.empty() &&
        !has_image_prompt && !options->batch_mode)
    {
        std::cerr << "--dflash requires --vlm-fixture or --image\n";
        return false;
    }
    return true;
}

} // namespace

CliParseResult parse_cli_options(int argc, char** argv, CliOptions* options)
{
    if (options == nullptr) return CliParseResult::ExitFailure;
    bool exit_success = false;
    if (parse_arguments(argc, argv, options, &exit_success)) return CliParseResult::Run;
    return exit_success ? CliParseResult::ExitSuccess : CliParseResult::ExitFailure;
}

} // namespace hunyuan_ocr::cli
