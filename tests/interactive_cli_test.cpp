#include "interactive_cli.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition) std::cerr << message << '\n';
    return condition;
}

} // namespace

int main()
{
    using hunyuan_ocr::PromptMode;
    using hunyuan_ocr::cli::InteractiveInputType;

    const auto quoted = hunyuan_ocr::cli::parse_interactive_input(
        "  \"/tmp/My Images/page 1.png\"  ");
    if (!expect(quoted.type == InteractiveInputType::Image, "quoted path was not parsed") ||
        !expect(quoted.value == "/tmp/My Images/page 1.png", "quoted path was not unwrapped"))
    {
        return 1;
    }

    const auto mode = hunyuan_ocr::cli::parse_interactive_input(":mode spotting");
    const auto prompt = hunyuan_ocr::cli::parse_interactive_input(":prompt visible text only");
    const auto reset = hunyuan_ocr::cli::parse_interactive_input(":prompt reset");
    const auto unknown = hunyuan_ocr::cli::parse_interactive_input(":backend vulkan");
    if (!expect(mode.type == InteractiveInputType::SetMode && mode.mode == PromptMode::Spotting,
                "spotting mode command was not parsed") ||
        !expect(prompt.type == InteractiveInputType::SetPrompt &&
                    prompt.value == "visible text only",
                "custom prompt command was not parsed") ||
        !expect(reset.type == InteractiveInputType::ResetPrompt,
                "prompt reset command was not parsed") ||
        !expect(unknown.type == InteractiveInputType::Error,
                "unknown command was not rejected"))
    {
        return 2;
    }

    hunyuan_ocr::cli::InteractiveSessionConfig config;
    config.model_root = "/models/hunyuan";
    config.backend_label = "CPU fp32";
    config.num_threads = 8;
    config.max_tokens = 8192;

    std::vector<std::string> images;
    std::vector<hunyuan_ocr::InferenceRequest> requests;
    const hunyuan_ocr::cli::InteractiveInfer infer =
        [&](const std::string& image,
            const hunyuan_ocr::InferenceRequest& request,
            hunyuan_ocr::InferenceResult* result,
            hunyuan_ocr::RuntimeError* error) {
            images.push_back(image);
            requests.push_back(request);
            if (image == "missing.png")
            {
                error->stage = "image_decode";
                error->message = "file was not found";
                return false;
            }
            request.stream_callback({123, "TEXT:" + image});
            result->timing.total_ms = 12.5;
            return true;
        };

    std::istringstream input(
        ":mode spotting\n"
        "\"first image.png\"\n"
        ":prompt visible text only\n"
        "missing.png\n"
        "second.png\n"
        ":prompt reset\n"
        ":status\n"
        ":quit\n");
    std::ostringstream output;
    std::ostringstream diagnostics;
    const int status = hunyuan_ocr::cli::run_interactive_session(
        config, infer, input, output, diagnostics);

    if (!expect(status == 0, "interactive session did not exit successfully") ||
        !expect(images.size() == 3, "interactive session did not process all image lines") ||
        !expect(images[0] == "first image.png", "session did not pass the unwrapped image path") ||
        !expect(requests[0].prompt_mode == PromptMode::Spotting,
                "mode command did not persist to the first request") ||
        !expect(requests[1].prompt_mode == PromptMode::Custom &&
                    requests[1].prompt == "visible text only",
                "custom prompt did not persist after one failed request") ||
        !expect(requests[2].prompt_mode == PromptMode::Custom &&
                    requests[2].prompt == "visible text only",
                "custom prompt did not persist to the next request") ||
        !expect(requests[2].max_tokens == 8192,
                "session max token setting was not propagated") ||
        !expect(output.str() == "TEXT:first image.png\nTEXT:second.png\n",
                "streamed OCR output was not written cleanly to stdout") ||
        !expect(diagnostics.str().find("Inference failed at image_decode: file was not found") !=
                    std::string::npos,
                "failed image did not produce an actionable error") ||
        !expect(diagnostics.str().find("Mode: document") != std::string::npos,
                "prompt reset did not restore document mode") ||
        !expect(diagnostics.str().find("Backend: CPU fp32") != std::string::npos,
                "status did not show the configured backend"))
    {
        return 3;
    }

    return 0;
}
