#include "hunyuan_ocr/vision_runtime.h"

#include <cmath>
#include <string>

int main()
{
    const float actual = hunyuan_ocr::detail::bilinear_source_coordinate(17, 128, 54);
    const float expected = (17.5f * 128.0f / 54.0f) - 0.5f;
    if (std::fabs(actual - expected) > 1e-6f) return 1;

    hunyuan_ocr::VisionRuntimeOptions options;
    options.use_vulkan = true;
    options.vulkan_device = 0;
    if (hunyuan_ocr::vision_vulkan_compiled()) return 0;

    hunyuan_ocr::VisionRuntime runtime(options);
    std::string error;
    if (runtime.load_dynamic("unused.ncnn.param",
                             "unused.ncnn.bin",
                             "unused.pos_embed.bin",
                             &error)) return 2;
    if (error != "ncnn was built without Vulkan support") return 3;
    return 0;
}
