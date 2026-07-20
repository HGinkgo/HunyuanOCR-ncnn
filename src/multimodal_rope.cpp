#include "hunyuan_ocr/multimodal_rope.h"

#include <algorithm>

#if NCNN_VULKAN
#include <layer_shader_type.h>
#include <pipeline.h>
#endif

namespace hunyuan_ocr {
namespace {

class MultimodalRotaryEmbed final : public ncnn::Layer {
public:
    MultimodalRotaryEmbed()
    {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
#if NCNN_VULKAN
        support_vulkan = true;
        support_vulkan_packing = true;
        pipeline_rotaryembed_ = nullptr;
        pipeline_rotaryembed_pack4_ = nullptr;
#endif
    }

    int load_param(const ncnn::ParamDict& pd) override
    {
        interleaved_ = pd.get(0, 0);
        full_cache_ = pd.get(1, 1);
        return 0;
    }

#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (!opt.use_vulkan_compute) return 0;
        if (vkdev == nullptr) return -1;

        const ncnn::Mat& shape = bottom_shapes.empty() ? ncnn::Mat() : bottom_shapes[0];
        std::vector<ncnn::vk_specialization_type> specializations(6);
        specializations[0].i = interleaved_;
        specializations[1].i = shape.w;
        specializations[2].i = shape.h;
        specializations[3].i = shape.c;
        specializations[4].i = static_cast<int>(shape.cstep);
        specializations[5].i = full_cache_;

        ncnn::Mat local_size_xyz;
        if (shape.dims == 3)
        {
            const int halfdim = shape.w / 2;
            local_size_xyz.w = std::min(64, std::max(1, halfdim));
            local_size_xyz.h = std::min(4, std::max(1, shape.h));
            local_size_xyz.c = std::min(4, std::max(1, shape.c));
        }
        else
        {
            local_size_xyz.w = 8;
            local_size_xyz.h = 8;
            local_size_xyz.c = 1;
        }

        if (shape.dims == 0 || shape.elempack == 1)
        {
            pipeline_rotaryembed_ = new ncnn::Pipeline(vkdev);
            pipeline_rotaryembed_->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_rotaryembed_->create(ncnn::LayerShaderType::rotaryembed, opt, specializations);
        }
        if (shape.dims == 0 || shape.elempack == 4)
        {
            pipeline_rotaryembed_pack4_ = new ncnn::Pipeline(vkdev);
            pipeline_rotaryembed_pack4_->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_rotaryembed_pack4_->create(ncnn::LayerShaderType::rotaryembed_pack4, opt, specializations);
        }
        return 0;
    }

    int destroy_pipeline(const ncnn::Option&) override
    {
        delete pipeline_rotaryembed_;
        pipeline_rotaryembed_ = nullptr;
        delete pipeline_rotaryembed_pack4_;
        pipeline_rotaryembed_pack4_ = nullptr;
        return 0;
    }

    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override
    {
        const ncnn::VkMat& bottom_blob = bottom_blobs[0];
        const ncnn::VkMat& cos_cache = bottom_blobs[1];
        const ncnn::VkMat& sin_cache = bottom_blobs[2];
        const int embed_dim = bottom_blob.w;
        const int seqlen = bottom_blob.h;
        const int num_heads = bottom_blob.c;
        const int elempack = bottom_blob.elempack;

        ncnn::VkMat cos_cache_unpacked = cos_cache;
        if (cos_cache.elempack != 1)
        {
            vkdev->convert_packing(cos_cache, cos_cache_unpacked, 1, cmd, opt);
            if (cos_cache_unpacked.empty()) return -100;
        }
        ncnn::VkMat sin_cache_unpacked = sin_cache;
        if (sin_cache.elempack != 1)
        {
            vkdev->convert_packing(sin_cache, sin_cache_unpacked, 1, cmd, opt);
            if (sin_cache_unpacked.empty()) return -100;
        }

        ncnn::VkMat& top_blob = top_blobs[0];
        top_blob.create_like(bottom_blob, opt.blob_vkallocator);
        if (top_blob.empty()) return -100;

        std::vector<ncnn::VkMat> bindings(4);
        bindings[0] = bottom_blob;
        bindings[1] = cos_cache_unpacked;
        bindings[2] = sin_cache_unpacked;
        bindings[3] = top_blob;
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].i = embed_dim;
        constants[1].i = seqlen;
        constants[2].i = num_heads;
        constants[3].i = static_cast<int>(bottom_blob.cstep);

        ncnn::VkMat dispatcher;
        dispatcher.w = embed_dim / 2;
        dispatcher.h = seqlen;
        dispatcher.c = num_heads;
        const ncnn::Pipeline* pipeline = elempack == 4 ? pipeline_rotaryembed_pack4_ : pipeline_rotaryembed_;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        return 0;
    }
#endif

    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override
    {
        const ncnn::Mat& input = bottom_blobs[0];
        const ncnn::Mat& cosine = bottom_blobs[1];
        const ncnn::Mat& sine = bottom_blobs[2];
        const int head_dim = input.w;
        const int seq_len = input.h;
        const int num_heads = input.c;

        ncnn::Mat& output = top_blobs[0];
        output.create_like(input, opt.blob_allocator);
        if (output.empty()) return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int head = 0; head < num_heads; ++head)
        {
            const ncnn::Mat input_head = input.channel(head);
            ncnn::Mat output_head = output.channel(head);
            for (int row = 0; row < seq_len; ++row)
            {
                const float* input_values = input_head.row(row);
                const float* cos_values = cosine.row(row);
                const float* sin_values = sine.row(row);
                float* output_values = output_head.row(row);
                if (interleaved_)
                {
                    for (int dim = 0; dim < head_dim; dim += 2)
                    {
                        const float first = input_values[dim];
                        const float second = input_values[dim + 1];
                        output_values[dim] = first * cos_values[dim / 2] - second * sin_values[dim / 2];
                        output_values[dim + 1] = first * sin_values[dim / 2] + second * cos_values[dim / 2];
                    }
                }
                else
                {
                    const int half_dim = head_dim / 2;
                    for (int dim = 0; dim < half_dim; ++dim)
                    {
                        const float first = input_values[dim];
                        const float second = input_values[dim + half_dim];
                        output_values[dim] = first * cos_values[dim] - second * sin_values[dim];
                        output_values[dim + half_dim] =
                            second * cos_values[dim + half_dim] + first * sin_values[dim + half_dim];
                    }
                }
            }
        }
        return 0;
    }

private:
    int interleaved_ = 0;
    int full_cache_ = 1;
#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_rotaryembed_ = nullptr;
    ncnn::Pipeline* pipeline_rotaryembed_pack4_ = nullptr;
#endif
};

} // namespace

ncnn::Layer* create_multimodal_rope_layer(void*)
{
    return new MultimodalRotaryEmbed;
}

} // namespace hunyuan_ocr
