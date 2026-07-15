#pragma once

#include <vector>

namespace ncnn {
class Layer;
}

namespace hunyuan_ocr::detail {

struct VisionVulkanPolicyResult {
    int gelu_layer_count = 0;
    int gelu_cpu_fallback_count = 0;
};

bool vision_vulkan_patchset_compiled();
VisionVulkanPolicyResult configure_exact_vision_gelu(
    const std::vector<ncnn::Layer*>& layers,
    bool exact_gelu_vulkan_available);

} // namespace hunyuan_ocr::detail
