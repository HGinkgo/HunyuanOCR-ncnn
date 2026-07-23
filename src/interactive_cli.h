#pragma once

#include "hunyuan_ocr/hunyuan_ocr.h"

#include <functional>
#include <iosfwd>
#include <string>

namespace hunyuan_ocr::cli {

enum class InteractiveInputType {
    Empty,
    Image,
    SetMode,
    SetPrompt,
    ResetPrompt,
    Status,
    Help,
    Quit,
    Error,
};

struct InteractiveInput {
    InteractiveInputType type = InteractiveInputType::Empty;
    std::string value;
    PromptMode mode = PromptMode::Document;
};

InteractiveInput parse_interactive_input(const std::string& line);

struct InteractiveSessionConfig {
    std::string model_root;
    std::string backend_label;
    int num_threads = 0;
    int max_tokens = 8192;
    PromptMode initial_mode = PromptMode::Document;
    std::string initial_prompt;
};

using InteractiveInfer = std::function<bool(const std::string&,
                                             const InferenceRequest&,
                                             InferenceResult*,
                                             RuntimeError*)>;

int run_interactive_session(const InteractiveSessionConfig& config,
                            const InteractiveInfer& infer,
                            std::istream& input,
                            std::ostream& output,
                            std::ostream& diagnostics);

} // namespace hunyuan_ocr::cli
