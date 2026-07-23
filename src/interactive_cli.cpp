#include "interactive_cli.h"

#include <iomanip>
#include <istream>
#include <ostream>
#include <string>

namespace hunyuan_ocr::cli {
namespace {

std::string trim(const std::string& text)
{
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string command_value(const std::string& text, const std::string& command)
{
    if (text.size() <= command.size()) return {};
    return trim(text.substr(command.size()));
}

void print_help(std::ostream& diagnostics)
{
    diagnostics << "Commands:\n"
                << "  :mode document|spotting  Set the built-in OCR prompt\n"
                << "  :prompt TEXT             Set a custom prompt\n"
                << "  :prompt reset            Restore document mode\n"
                << "  :status                  Show the current session settings\n"
                << "  :help                    Show this help\n"
                << "  :quit, :exit             Exit\n";
}

void print_status(const InteractiveSessionConfig& config,
                  PromptMode mode,
                  const std::string& prompt,
                  std::ostream& diagnostics)
{
    diagnostics << "Model: " << config.model_root << '\n'
                << "Backend: " << config.backend_label << '\n'
                << "Threads: " << config.num_threads << '\n'
                << "Mode: " << prompt_mode_name(mode) << '\n';
    if (mode == PromptMode::Custom) diagnostics << "Prompt: " << prompt << '\n';
}

} // namespace

InteractiveInput parse_interactive_input(const std::string& line)
{
    const std::string text = trim(line);
    if (text.empty()) return {};

    if (text.front() != ':')
    {
        InteractiveInput result;
        result.type = InteractiveInputType::Image;
        result.value = text;
        if (result.value.size() >= 2 &&
            ((result.value.front() == '"' && result.value.back() == '"') ||
             (result.value.front() == '\'' && result.value.back() == '\'')))
        {
            result.value = result.value.substr(1, result.value.size() - 2);
        }
        return result;
    }

    InteractiveInput result;
    if (text == ":quit" || text == ":exit")
    {
        result.type = InteractiveInputType::Quit;
        return result;
    }
    if (text == ":help")
    {
        result.type = InteractiveInputType::Help;
        return result;
    }
    if (text == ":status")
    {
        result.type = InteractiveInputType::Status;
        return result;
    }
    if (text.rfind(":mode", 0) == 0)
    {
        const std::string value = command_value(text, ":mode");
        if (value == "document")
        {
            result.type = InteractiveInputType::SetMode;
            result.mode = PromptMode::Document;
            return result;
        }
        if (value == "spotting")
        {
            result.type = InteractiveInputType::SetMode;
            result.mode = PromptMode::Spotting;
            return result;
        }
        result.type = InteractiveInputType::Error;
        result.value = "Usage: :mode document|spotting";
        return result;
    }
    if (text.rfind(":prompt", 0) == 0)
    {
        const std::string value = command_value(text, ":prompt");
        if (value == "reset")
        {
            result.type = InteractiveInputType::ResetPrompt;
            return result;
        }
        if (!value.empty())
        {
            result.type = InteractiveInputType::SetPrompt;
            result.value = value;
            return result;
        }
        result.type = InteractiveInputType::Error;
        result.value = "Usage: :prompt TEXT|reset";
        return result;
    }

    result.type = InteractiveInputType::Error;
    result.value = "Unknown command. Type :help for available commands.";
    return result;
}

int run_interactive_session(const InteractiveSessionConfig& config,
                            const InteractiveInfer& infer,
                            std::istream& input,
                            std::ostream& output,
                            std::ostream& diagnostics)
{
    PromptMode mode = config.initial_prompt.empty() ? config.initial_mode : PromptMode::Custom;
    std::string prompt = config.initial_prompt;

    diagnostics << "Ready. Enter an image path or :help.\n";
    std::string line;
    while (true)
    {
        diagnostics << "Image> " << std::flush;
        if (!std::getline(input, line))
        {
            diagnostics << '\n';
            return 0;
        }

        const InteractiveInput command = parse_interactive_input(line);
        switch (command.type)
        {
        case InteractiveInputType::Empty:
            continue;
        case InteractiveInputType::Quit:
            return 0;
        case InteractiveInputType::Help:
            print_help(diagnostics);
            continue;
        case InteractiveInputType::Status:
            print_status(config, mode, prompt, diagnostics);
            continue;
        case InteractiveInputType::SetMode:
            mode = command.mode;
            prompt.clear();
            diagnostics << "Mode: " << prompt_mode_name(mode) << '\n';
            continue;
        case InteractiveInputType::SetPrompt:
            mode = PromptMode::Custom;
            prompt = command.value;
            diagnostics << "Mode: custom\n";
            continue;
        case InteractiveInputType::ResetPrompt:
            mode = PromptMode::Document;
            prompt.clear();
            diagnostics << "Mode: document\n";
            continue;
        case InteractiveInputType::Error:
            diagnostics << command.value << '\n';
            continue;
        case InteractiveInputType::Image:
            break;
        }

        InferenceRequest request;
        request.prompt_mode = mode;
        request.prompt = prompt;
        request.max_tokens = config.max_tokens;
        request.stream_callback = [&output](const InferenceChunk& chunk) {
            if (!chunk.text_delta.empty()) output << chunk.text_delta << std::flush;
        };

        diagnostics << "Recognizing: " << command.value << '\n';
        InferenceResult result;
        RuntimeError error;
        if (!infer(command.value, request, &result, &error))
        {
            diagnostics << "Inference failed at " << error.stage << ": " << error.message << '\n';
            continue;
        }

        output << '\n';
        diagnostics << std::fixed << std::setprecision(1)
                    << "Done in " << result.timing.total_ms / 1000.0 << " s.\n";
    }
}

} // namespace hunyuan_ocr::cli
