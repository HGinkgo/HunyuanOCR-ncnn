#include "hunyuan_ocr/precise_sdpa.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace hunyuan_ocr {
namespace {

class PreciseSDPA final : public ncnn::Layer {
public:
    PreciseSDPA()
    {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
    }

    int load_param(const ncnn::ParamDict& pd) override
    {
        attn_mask_ = pd.get(5, 0);
        scale_ = pd.get(6, 0.0f);
        kv_cache_ = pd.get(7, 0);
        return 0;
    }

    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override
    {
        const ncnn::Mat& query = bottom_blobs[0];
        const ncnn::Mat& current_key = bottom_blobs[1];
        const ncnn::Mat& current_value = bottom_blobs[2];
        const ncnn::Mat& mask = attn_mask_ ? bottom_blobs[3] : ncnn::Mat();
        const ncnn::Mat& past_key = kv_cache_ ? bottom_blobs[attn_mask_ ? 4 : 3] : ncnn::Mat();
        const ncnn::Mat& past_value = kv_cache_ ? bottom_blobs[attn_mask_ ? 5 : 4] : ncnn::Mat();

        const int head_dim = query.w;
        const int query_len = query.h;
        const int num_heads = query.c;
        const int current_len = current_key.h;
        const int num_kv_heads = current_key.c;
        const int value_dim = current_value.w;
        const int past_len = kv_cache_ ? past_key.h : 0;
        const int key_len = past_len + current_len;
        const int heads_per_group = num_heads / num_kv_heads;
        const double scale = scale_ == 0.0f ? 1.0 / std::sqrt(static_cast<double>(head_dim)) : scale_;

        ncnn::Mat key = current_key;
        ncnn::Mat value = current_value;
        if (past_len > 0)
        {
            key.create(head_dim, key_len, num_kv_heads, 4u, opt.blob_allocator);
            value.create(value_dim, key_len, num_kv_heads, 4u, opt.blob_allocator);
            if (key.empty() || value.empty()) return -100;
            for (int head = 0; head < num_kv_heads; ++head)
            {
                std::memcpy(key.channel(head).row(0), past_key.channel(head),
                            static_cast<size_t>(head_dim) * past_len * sizeof(float));
                std::memcpy(key.channel(head).row(past_len), current_key.channel(head),
                            static_cast<size_t>(head_dim) * current_len * sizeof(float));
                std::memcpy(value.channel(head).row(0), past_value.channel(head),
                            static_cast<size_t>(value_dim) * past_len * sizeof(float));
                std::memcpy(value.channel(head).row(past_len), current_value.channel(head),
                            static_cast<size_t>(value_dim) * current_len * sizeof(float));
            }
        }

        ncnn::Mat& output = top_blobs[0];
        output.create(value_dim, query_len, num_heads, 4u, opt.blob_allocator);
        if (output.empty()) return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int head = 0; head < num_heads; ++head)
        {
            const ncnn::Mat query_head = query.channel(head);
            const ncnn::Mat key_head = key.channel(head / heads_per_group);
            const ncnn::Mat value_head = value.channel(head / heads_per_group);
            const ncnn::Mat mask_head = attn_mask_ && mask.c > 1 ? mask.channel(head) : mask;
            ncnn::Mat output_head = output.channel(head);
            std::vector<double> probabilities(static_cast<size_t>(key_len));

            for (int row = 0; row < query_len; ++row)
            {
                const float* query_values = query_head.row(row);
                double maximum = -std::numeric_limits<double>::infinity();
                for (int key_index = 0; key_index < key_len; ++key_index)
                {
                    const float* key_values = key_head.row(key_index);
                    double score = 0.0;
                    for (int dim = 0; dim < head_dim; ++dim)
                    {
                        score += static_cast<double>(query_values[dim]) * key_values[dim];
                    }
                    score *= scale;
                    if (attn_mask_) score += mask_head.row(row)[key_index];
                    probabilities[static_cast<size_t>(key_index)] = score;
                    maximum = std::max(maximum, score);
                }

                double denominator = 0.0;
                for (double& probability : probabilities)
                {
                    probability = std::exp(probability - maximum);
                    denominator += probability;
                }
                for (double& probability : probabilities) probability /= denominator;

                float* output_values = output_head.row(row);
                for (int dim = 0; dim < value_dim; ++dim)
                {
                    double sum = 0.0;
                    for (int key_index = 0; key_index < key_len; ++key_index)
                    {
                        sum += probabilities[static_cast<size_t>(key_index)] * value_head.row(key_index)[dim];
                    }
                    output_values[dim] = static_cast<float>(sum);
                }
            }
        }

        if (kv_cache_)
        {
            top_blobs[1] = key;
            top_blobs[2] = value;
        }
        return 0;
    }

private:
    int attn_mask_ = 0;
    float scale_ = 0.0f;
    int kv_cache_ = 0;
};

} // namespace

ncnn::Layer* create_precise_sdpa_layer(void*)
{
    return new PreciseSDPA;
}

} // namespace hunyuan_ocr
