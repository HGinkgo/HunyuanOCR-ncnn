#pragma once

namespace hunyuan_ocr {
namespace detail {

enum class TextNetStage
{
    Embedding,
    Decoder,
    LmHead,
};

inline bool text_stage_uses_vulkan(TextNetStage stage, bool text_vulkan_requested)
{
    return text_vulkan_requested && stage != TextNetStage::Embedding;
}

} // namespace detail
} // namespace hunyuan_ocr
