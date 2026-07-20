#include "text_decoder_step.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#if NCNN_VULKAN
#include <gpu.h>
#endif

namespace {

#if NCNN_VULKAN

std::string make_decoder_param()
{
    std::ostringstream param;
    param << "7767517\n55 103\n";
    for (int index = 0; index < 6; ++index)
    {
        param << "Input in" << index << " 0 1 in" << index << '\n';
    }
    param << "Input kv_cache 0 48";
    for (int layer = 0; layer < hunyuan_ocr::detail::kTextAttentionLayerCount; ++layer)
    {
        param << " cache_k" << layer << " cache_v" << layer;
    }
    param << '\n';
    for (int layer = 0; layer < hunyuan_ocr::detail::kTextAttentionLayerCount; ++layer)
    {
        if (layer == 0)
        {
            param << "Split key0 1 2 cache_k0 out0 out_cache_k0\n";
        }
        else
        {
            param << "Split key" << layer << " 1 1 cache_k" << layer
                  << " out_cache_k" << layer << '\n';
        }
        param << "Split value" << layer << " 1 1 cache_v" << layer
              << " out_cache_v" << layer << '\n';
    }
    return param.str();
}

ncnn::Mat filled_mat(float value)
{
    ncnn::Mat result(4, 1, 1);
    result.fill(value);
    return result;
}

int run_vulkan_test()
{
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.set_vulkan_device(0);
    const std::string param = make_decoder_param();
    if (net.load_param_mem(param.c_str()) != 0) return 1;

    hunyuan_ocr::detail::DecoderKVCache caches;
    for (int layer = 0; layer < hunyuan_ocr::detail::kTextAttentionLayerCount; ++layer)
    {
        caches.emplace_back(filled_mat(10.0f + layer), filled_mat(20.0f + layer));
    }

    std::string error;
    hunyuan_ocr::detail::VulkanDecoderSession session(net);
    if (!session.initialize(caches, &error)) return 2;
    caches[0].first.fill(-999.0f);

    const ncnn::Mat input = filled_mat(1.0f);
    const ncnn::Mat mask = filled_mat(0.0f);
    const ncnn::Mat cos = filled_mat(1.0f);
    const ncnn::Mat sin = filled_mat(0.0f);
    for (int step = 0; step < 2; ++step)
    {
        ncnn::Mat hidden;
        if (!session.run_step(input, mask, cos, sin, &hidden, &error))
        {
            std::cerr << error << '\n';
            return 3;
        }
        if (hidden.total() != 4 || std::fabs(hidden[0] - 10.0f) > 1e-6f) return 4;
    }
    if (session.step_submit_count() != 2) return 5;
    return 0;
}

#endif

} // namespace

int main()
{
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() == 0)
    {
        ncnn::destroy_gpu_instance();
        return 0;
    }
    const int result = run_vulkan_test();
    ncnn::destroy_gpu_instance();
    return result;
#else
    return 0;
#endif
}
