#include "hunyuan_ocr/multimodal_rope.h"

namespace hunyuan_ocr {
namespace {

class MultimodalRotaryEmbed final : public ncnn::Layer {
public:
    MultimodalRotaryEmbed()
    {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
    }

    int load_param(const ncnn::ParamDict& pd) override
    {
        interleaved_ = pd.get(0, 0);
        return 0;
    }

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
};

} // namespace

ncnn::Layer* create_multimodal_rope_layer(void*)
{
    return new MultimodalRotaryEmbed;
}

} // namespace hunyuan_ocr
