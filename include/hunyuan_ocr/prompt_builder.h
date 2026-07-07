#pragma once

#include <string>
#include <vector>

namespace hunyuan_ocr {

enum class PromptMode {
    Spotting,
    Document,
};

struct PromptBuildResult {
    std::vector<int> chat_template_ids;
    std::vector<int> input_ids;
    std::vector<int> position_ids;
    int seq_len = 0;
    int vision_token_count = 0;
    int image_token_id = 120120;
};

bool parse_prompt_mode(const std::string& text, PromptMode* mode, std::string* error);
const char* prompt_mode_name(PromptMode mode);

bool build_hunyuan_ocr_prompt(PromptMode mode,
                              int grid_h,
                              int grid_w,
                              int merge_size,
                              PromptBuildResult* result,
                              std::string* error);

bool build_hunyuan_ocr_prompt_from_tokens(const std::vector<int>& prompt_token_ids,
                                          int grid_h,
                                          int grid_w,
                                          int merge_size,
                                          PromptBuildResult* result,
                                          std::string* error);

} // namespace hunyuan_ocr
