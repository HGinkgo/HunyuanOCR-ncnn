#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/utf8.h"

#include "batch_jsonl.h"
#include "cli_benchmark.h"
#include "cli_fixture.h"
#include "cli_options.h"
#include "interactive_cli.h"

#include <chrono>
#include <iomanip>
#include <iostream>
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

void print_file_group(std::ostream& output,
                      const char* title,
                      const std::vector<hunyuan_ocr::ModelFile>& files)
{
    output << title << "\n";
    if (files.empty())
    {
        output << "  (none)\n";
        return;
    }

    for (const auto& file : files)
    {
        output << "  " << file.relative_path << " - " << file.note << "\n";
    }
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
    hunyuan_ocr::cli::CliOptions options;
    const hunyuan_ocr::cli::CliParseResult parse_result =
        hunyuan_ocr::cli::parse_cli_options(argc, argv, &options);
    if (parse_result == hunyuan_ocr::cli::CliParseResult::ExitSuccess) return 0;
    if (parse_result == hunyuan_ocr::cli::CliParseResult::ExitFailure) return 1;

    const std::string& model_root = options.model_root;
    const bool dflash = options.dflash;
    const bool mmap_weights = options.mmap_weights;
    const bool interactive = options.interactive;
    const bool vulkan = options.vulkan;
    const std::string& image_path = options.image_path;
    const std::string& batch_input_path = options.batch_input_path;
    const std::string& batch_output_path = options.batch_output_path;
    const bool force_batch_output = options.force_batch_output;
    const std::string& prompt_mode_text = options.prompt_mode_text;
    const std::string& prompt_text = options.prompt_text;
    const int max_tokens = options.max_tokens;
    const float repetition_penalty = options.repetition_penalty;
    const bool benchmark = options.benchmark;
    const int benchmark_warmup = options.benchmark_warmup;
    const int benchmark_repeat = options.benchmark_repeat;
    const int num_threads = options.num_threads;
    const bool vision_vulkan = options.vision_vulkan;
    const int vision_vulkan_device = options.vision_vulkan_device;
    const bool text_vulkan = options.text_vulkan;
    const int text_vulkan_device = options.text_vulkan_device;
    const bool batch_mode = options.batch_mode;
    const bool plain_image_inference = options.plain_image_inference;
    const hunyuan_ocr::VisionRuntimeOptions& vision_options = options.vision_options;

    const hunyuan_ocr::ModelLayoutReport report = hunyuan_ocr::check_model_layout(model_root);
    const bool ready = report.required_files_present();

    if (!benchmark)
    {
        if (plain_image_inference || interactive)
        {
            std::cerr << "Loading OCR model from " << report.root << "\n";
        }
        else
        {
            std::cout << "Model root: " << report.root << "\n";
            print_file_group(std::cout, "Present files:", report.present);
            print_file_group(std::cout, "Missing required files:", report.missing_required);
            print_file_group(std::cout, "Missing planned files:", report.missing_planned);
        }
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
        runtime_options.text_vulkan = text_vulkan;
        runtime_options.text_vulkan_device = text_vulkan_device;
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
        return hunyuan_ocr::cli::run_image_benchmark(model_root,
                                                      image_path,
                                                      prompt_mode_text,
                                                      prompt_text,
                                                      max_tokens,
                                                      repetition_penalty,
                                                      num_threads,
                                                      benchmark_warmup,
                                                      benchmark_repeat,
                                                      vision_options,
                                                      text_vulkan,
                                                      text_vulkan_device,
                                                      process_start);
    }

    if (!plain_image_inference && !interactive)
    {
        std::cout << "Model layout check passed for the current runtime files.\n";
    }

    if (interactive)
    {
        hunyuan_ocr::RuntimeOptions runtime_options;
        runtime_options.num_threads = num_threads;
        runtime_options.vision_vulkan = vision_vulkan;
        runtime_options.vision_vulkan_device = vision_vulkan_device;
        runtime_options.dflash = dflash;
        runtime_options.mmap_weights = mmap_weights;
        runtime_options.text_vulkan = text_vulkan;
        runtime_options.text_vulkan_device = text_vulkan_device;
        runtime_options.repetition_penalty = repetition_penalty;

        hunyuan_ocr::PromptMode initial_mode = hunyuan_ocr::PromptMode::Document;
        if (!prompt_mode_text.empty())
        {
            std::string prompt_error;
            if (!hunyuan_ocr::parse_prompt_mode(prompt_mode_text, &initial_mode, &prompt_error))
            {
                std::cerr << "Prompt mode failed: " << prompt_error << "\n";
                return 1;
            }
        }

        hunyuan_ocr::RuntimeError runtime_error;
        hunyuan_ocr::HunyuanOCR runtime;
        const auto load_start = Clock::now();
        if (!runtime.load(model_root, runtime_options, &runtime_error))
        {
            std::cerr << "Runtime load failed at " << runtime_error.stage
                      << ": " << runtime_error.message << "\n";
            return 1;
        }
        std::cerr << std::fixed << std::setprecision(1)
                  << "Model ready in " << elapsed_ms(load_start, Clock::now()) / 1000.0
                  << " s.\n";

        hunyuan_ocr::cli::InteractiveSessionConfig config;
        config.model_root = model_root;
        config.backend_label = vulkan ? "Vulkan fp32" : "CPU fp32";
        config.num_threads = num_threads;
        config.max_tokens = max_tokens > 0
            ? max_tokens
            : hunyuan_ocr::cli::kDefaultImageMaxTokens;
        config.initial_mode = initial_mode;
        config.initial_prompt = prompt_text;
        return hunyuan_ocr::cli::run_interactive_session(
            config,
            [&](const std::string& image,
                const hunyuan_ocr::InferenceRequest& request,
                hunyuan_ocr::InferenceResult* result,
                hunyuan_ocr::RuntimeError* error) {
                return runtime.infer_file(image, request, result, error);
            },
            std::cin,
            std::cout,
            std::cerr);
    }

    if (plain_image_inference)
    {
        hunyuan_ocr::RuntimeOptions runtime_options;
        runtime_options.num_threads = num_threads;
        runtime_options.vision_vulkan = vision_vulkan;
        runtime_options.vision_vulkan_device = vision_vulkan_device;
        runtime_options.dflash = dflash;
        runtime_options.mmap_weights = mmap_weights;
        runtime_options.text_vulkan = text_vulkan;
        runtime_options.text_vulkan_device = text_vulkan_device;
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
        request.max_tokens = max_tokens > 0
            ? max_tokens
            : hunyuan_ocr::cli::kDefaultImageMaxTokens;
        if (!prompt_text.empty())
        {
            request.prompt_mode = hunyuan_ocr::PromptMode::Custom;
            request.prompt = prompt_text;
        }
        else if (!prompt_mode_text.empty())
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
        else
        {
            request.prompt_mode = hunyuan_ocr::PromptMode::Document;
        }

        request.stream_callback = [](const hunyuan_ocr::InferenceChunk& chunk) {
            if (!chunk.text_delta.empty())
            {
                std::cout << chunk.text_delta << std::flush;
            }
        };

        std::cerr << "Loading image: " << image_path << "\n";
        std::cerr << "Generating text:\n";

        hunyuan_ocr::InferenceResult result;
        if (!runtime.infer_file(image_path, request, &result, &runtime_error))
        {
            std::cerr << "Inference failed at " << runtime_error.stage
                      << ": " << runtime_error.message << "\n";
            return 1;
        }

        std::cerr << "\n\nDone.\n";
        std::cerr << "Inference:\n";
        std::cerr << "  prompt_mode: "
                  << hunyuan_ocr::prompt_mode_name(request.prompt_mode) << "\n";
        std::cerr << "  decoder: "
                  << (result.decoder.mode == hunyuan_ocr::DecoderMode::DFlash ? "dflash" : "ar")
                  << "\n";
        std::cerr << "  generated_token_count: " << result.token_ids.size() << "\n";
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "  timing_preprocess_ms: " << result.timing.preprocess_ms << "\n";
        std::cerr << "  timing_vision_ms: " << result.timing.vision_ms << "\n";
        std::cerr << "  timing_text_ms: " << result.timing.text_ms << "\n";
        std::cerr << "  timing_total_ms: " << result.timing.total_ms << "\n";
        if (result.decoder.mode == hunyuan_ocr::DecoderMode::DFlash)
        {
            std::cerr << "  dflash_blocks: " << result.decoder.block_count << "\n";
            std::cerr << "  dflash_drafted_tokens: "
                      << result.decoder.drafted_token_count << "\n";
            std::cerr << "  dflash_accepted_tokens: "
                      << result.decoder.accepted_draft_token_count << "\n";
        }
        return 0;
    }

    return hunyuan_ocr::cli::run_fixture_mode(options);
}
