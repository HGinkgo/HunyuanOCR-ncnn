#include "hunyuan_ocr/prompt_builder.h"

#include <algorithm>

namespace hunyuan_ocr {
namespace {

constexpr int kImageTokenId = 120120;

const int kSpottingTemplateIds[] = {
    120000, 120021, 120118, 120120, 120119, 5055, 951, 9977, 12858, 1843,
    9738, 270, 934, 8433, 4699, 68216, 8287, 292, 120006,
};

const int kDocumentTemplateIds[] = {
    120000, 120021, 120118, 120120, 120119, 12161, 19177, 12858, 409, 49827,
    16317, 1940, 474, 78338, 11971, 2598, 270, 2627, 6319, 19684, 332, 6319,
    5705, 2218, 19664, 270, 24784, 474, 13110, 11971, 4088, 270, 19177, 409,
    4605, 474, 36078, 11971, 2598, 270, 3802, 6169, 10264, 2297, 1170, 4134,
    292, 120006,
};

std::vector<int> template_ids_for_mode(PromptMode mode)
{
    if (mode == PromptMode::Spotting)
    {
        return std::vector<int>(std::begin(kSpottingTemplateIds), std::end(kSpottingTemplateIds));
    }
    return std::vector<int>(std::begin(kDocumentTemplateIds), std::end(kDocumentTemplateIds));
}

} // namespace

bool parse_prompt_mode(const std::string& text, PromptMode* mode, std::string* error)
{
    if (mode == nullptr)
    {
        if (error) *error = "prompt mode pointer is null";
        return false;
    }
    if (text == "spotting")
    {
        *mode = PromptMode::Spotting;
        return true;
    }
    if (text == "document")
    {
        *mode = PromptMode::Document;
        return true;
    }
    if (error) *error = "unsupported prompt mode: " + text + " (expected spotting or document)";
    return false;
}

const char* prompt_mode_name(PromptMode mode)
{
    return mode == PromptMode::Spotting ? "spotting" : "document";
}

bool build_hunyuan_ocr_prompt(PromptMode mode,
                              int grid_h,
                              int grid_w,
                              int merge_size,
                              PromptBuildResult* result,
                              std::string* error)
{
    if (result == nullptr)
    {
        if (error) *error = "prompt build result pointer is null";
        return false;
    }
    if (grid_h <= 0 || grid_w <= 0 || merge_size <= 0)
    {
        if (error) *error = "grid and merge_size must be positive";
        return false;
    }
    if (grid_h % merge_size != 0 || grid_w % merge_size != 0)
    {
        if (error) *error = "grid dimensions must be divisible by merge_size";
        return false;
    }

    const int patch_h = grid_h / merge_size;
    const int patch_w = grid_w / merge_size;
    const int spatial_token_count = patch_h * (patch_w + 1);
    const int vision_token_count = spatial_token_count + 2;

    PromptBuildResult local;
    local.chat_template_ids = template_ids_for_mode(mode);
    local.vision_token_count = vision_token_count;
    local.image_token_id = kImageTokenId;

    const int image_token_templates = static_cast<int>(
        std::count(local.chat_template_ids.begin(), local.chat_template_ids.end(), kImageTokenId));
    if (image_token_templates != 1)
    {
        if (error) *error = "built-in prompt template must contain exactly one image token";
        return false;
    }

    local.input_ids.reserve(local.chat_template_ids.size() + static_cast<size_t>(vision_token_count));
    int first_image_pos = -1;
    for (const int token : local.chat_template_ids)
    {
        if (token != kImageTokenId)
        {
            local.input_ids.push_back(token);
            continue;
        }
        first_image_pos = static_cast<int>(local.input_ids.size());
        for (int i = 0; i < vision_token_count; ++i)
        {
            local.input_ids.push_back(kImageTokenId);
        }
    }

    local.seq_len = static_cast<int>(local.input_ids.size());
    local.position_ids.assign(static_cast<size_t>(local.seq_len) * 4, 0);
    for (int axis = 0; axis < 4; ++axis)
    {
        for (int pos = 0; pos < local.seq_len; ++pos)
        {
            local.position_ids[static_cast<size_t>(axis) * local.seq_len + pos] = pos;
        }
    }

    const int spatial_start = first_image_pos + 1;
    if (spatial_start < 0 || spatial_start + spatial_token_count > local.seq_len)
    {
        if (error) *error = "expanded image token range is invalid";
        return false;
    }
    for (int h = 0; h < patch_h; ++h)
    {
        for (int w = 0; w < patch_w + 1; ++w)
        {
            const int offset = h * (patch_w + 1) + w;
            const int pos = spatial_start + offset;
            local.position_ids[static_cast<size_t>(1) * local.seq_len + pos] = w;
            local.position_ids[static_cast<size_t>(2) * local.seq_len + pos] = h;
            local.position_ids[static_cast<size_t>(3) * local.seq_len + pos] = 0;
        }
    }

    *result = std::move(local);
    return true;
}

} // namespace hunyuan_ocr
