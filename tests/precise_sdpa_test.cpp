#include "hunyuan_ocr/precise_sdpa.h"

#include <cmath>
#include <memory>
#include <vector>

int main()
{
    std::unique_ptr<ncnn::Layer> layer(hunyuan_ocr::create_precise_sdpa_layer(nullptr));
    ncnn::ParamDict params;
    params.set(5, 0);
    params.set(6, 1.0f);
    params.set(7, 0);
    if (layer->load_param(params) != 0)
    {
        return 1;
    }

    ncnn::Mat query(3, 1, 1);
    ncnn::Mat key(3, 2, 1);
    ncnn::Mat value(1, 2, 1);
    float* q = query.channel(0).row(0);
    q[0] = 1.0e8f;
    q[1] = 1.0f;
    q[2] = -1.0e8f;
    float* k0 = key.channel(0).row(0);
    k0[0] = 1.0f;
    k0[1] = 1.0f;
    k0[2] = 1.0f;
    key.channel(0).row(1)[0] = 0.0f;
    key.channel(0).row(1)[1] = 0.0f;
    key.channel(0).row(1)[2] = 0.0f;
    value.channel(0).row(0)[0] = 10.0f;
    value.channel(0).row(1)[0] = 20.0f;

    std::vector<ncnn::Mat> outputs(1);
    ncnn::Option options;
    options.num_threads = 1;
    if (layer->forward({query, key, value}, outputs, options) != 0)
    {
        return 2;
    }

    const double probability = std::exp(1.0) / (std::exp(1.0) + 1.0);
    const float expected = static_cast<float>(probability * 10.0 + (1.0 - probability) * 20.0);
    return std::fabs(outputs[0][0] - expected) <= 1e-5f ? 0 : 3;
}
