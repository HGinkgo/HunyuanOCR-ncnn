#include "vision_vulkan_policy.h"

#include <layer.h>
#include <layer_type.h>

#include <memory>
#include <vector>

int main()
{
    std::vector<std::unique_ptr<ncnn::Layer>> storage;
    std::vector<ncnn::Layer*> layers;
    for (int i = 0; i < 28; ++i)
    {
        storage.emplace_back(new ncnn::Layer);
        ncnn::Layer* layer = storage.back().get();
        layer->typeindex = ncnn::LayerType::GELU;
        layer->support_vulkan = true;
        layers.push_back(layer);
    }

    storage.emplace_back(new ncnn::Layer);
    ncnn::Layer* non_gelu = storage.back().get();
    non_gelu->typeindex = ncnn::LayerType::ReLU;
    non_gelu->support_vulkan = true;
    layers.push_back(non_gelu);

    const auto stock = hunyuan_ocr::detail::configure_exact_vision_gelu(layers, false);
    if (stock.gelu_layer_count != 28 || stock.gelu_cpu_fallback_count != 28) return 1;
    for (int i = 0; i < 28; ++i)
    {
        if (layers[static_cast<size_t>(i)]->support_vulkan) return 2;
    }
    if (!non_gelu->support_vulkan) return 3;

    for (int i = 0; i < 28; ++i)
    {
        layers[static_cast<size_t>(i)]->support_vulkan = true;
    }
    const auto patched = hunyuan_ocr::detail::configure_exact_vision_gelu(layers, true);
    if (patched.gelu_layer_count != 28 || patched.gelu_cpu_fallback_count != 0) return 4;
    for (int i = 0; i < 28; ++i)
    {
        if (!layers[static_cast<size_t>(i)]->support_vulkan) return 5;
    }
    if (!non_gelu->support_vulkan) return 6;
    return 0;
}
