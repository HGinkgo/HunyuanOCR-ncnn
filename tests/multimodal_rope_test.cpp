#include "hunyuan_ocr/multimodal_rope.h"

#include <cmath>
#include <memory>
#include <vector>

int main()
{
    std::unique_ptr<ncnn::Layer> layer(hunyuan_ocr::create_multimodal_rope_layer(nullptr));
#if NCNN_VULKAN
    if (!layer->support_vulkan) return 4;
#endif
    ncnn::ParamDict params;
    params.set(0, 0);
    if (layer->load_param(params) != 0) return 1;

    ncnn::Mat input(4, 1, 1);
    ncnn::Mat cosine(4, 1);
    ncnn::Mat sine(4, 1);
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = 3.0f;
    input[3] = 4.0f;
    cosine[0] = 0.5f;
    cosine[1] = 0.25f;
    cosine[2] = 0.75f;
    cosine[3] = 0.125f;
    sine[0] = 0.1f;
    sine[1] = 0.2f;
    sine[2] = 0.3f;
    sine[3] = 0.4f;

    std::vector<ncnn::Mat> outputs(1);
    ncnn::Option options;
    options.num_threads = 1;
    if (layer->create_pipeline(options) != 0) return 5;
    if (layer->forward({input, cosine, sine}, outputs, options) != 0) return 2;

    const float expected[4] = {
        1.0f * 0.5f - 3.0f * 0.1f,
        2.0f * 0.25f - 4.0f * 0.2f,
        3.0f * 0.75f + 1.0f * 0.3f,
        4.0f * 0.125f + 2.0f * 0.4f,
    };
    for (int index = 0; index < 4; ++index)
    {
        if (std::fabs(outputs[0][index] - expected[index]) > 1e-6f) return 3;
    }
    if (layer->destroy_pipeline(options) != 0) return 6;
    return 0;
}
