#include "hunyuan_ocr/precise_sdpa.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace {

bool near(float value, float expected, float tol = 1e-5f)
{
    return std::fabs(value - expected) <= tol;
}

void fill_channel_rows(ncnn::Mat& mat, float seed)
{
    for (int c = 0; c < mat.c; ++c)
    {
        for (int h = 0; h < mat.h; ++h)
        {
            float* row = mat.channel(c).row(h);
            for (int w = 0; w < mat.w; ++w)
            {
                row[w] = seed + 0.01f * static_cast<float>(c) +
                         0.1f * static_cast<float>(h) + 0.001f * static_cast<float>(w);
            }
        }
    }
}

bool mats_equal(const ncnn::Mat& left, const ncnn::Mat& right, float tol = 0.0f)
{
    if (left.dims != right.dims || left.w != right.w || left.h != right.h ||
        left.c != right.c || left.elemsize != right.elemsize ||
        left.elempack != right.elempack)
    {
        return false;
    }
    for (int c = 0; c < left.c; ++c)
    {
        for (int h = 0; h < left.h; ++h)
        {
            const float* a = left.channel(c).row(h);
            const float* b = right.channel(c).row(h);
            for (int w = 0; w < left.w; ++w)
            {
                if (std::fabs(a[w] - b[w]) > tol)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

int fail(const char* message, int code)
{
    std::cerr << message << '\n';
    return code;
}

// Reference attention without KV cache, used to validate cache-path math.
int reference_attention(const ncnn::Mat& query,
                        const ncnn::Mat& key,
                        const ncnn::Mat& value,
                        float scale,
                        ncnn::Mat* output)
{
    std::unique_ptr<ncnn::Layer> layer(hunyuan_ocr::create_precise_sdpa_layer(nullptr));
    ncnn::ParamDict params;
    params.set(5, 0); // no mask
    params.set(6, scale);
    params.set(7, 0); // no kv cache
    if (layer->load_param(params) != 0)
    {
        return 1;
    }
    std::vector<ncnn::Mat> outs(1);
    ncnn::Option options;
    options.num_threads = 1;
    if (layer->forward({query, key, value}, outs, options) != 0)
    {
        return 2;
    }
    *output = outs[0];
    return 0;
}

} // namespace

int main()
{
    // Case 1: original no-cache numerical smoke.
    {
        std::unique_ptr<ncnn::Layer> layer(hunyuan_ocr::create_precise_sdpa_layer(nullptr));
        ncnn::ParamDict params;
        params.set(5, 0);
        params.set(6, 1.0f);
        params.set(7, 0);
        if (layer->load_param(params) != 0)
        {
            return fail("load_param no-cache failed", 1);
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
            return fail("no-cache forward failed", 2);
        }

        const double probability = std::exp(1.0) / (std::exp(1.0) + 1.0);
        const float expected = static_cast<float>(probability * 10.0 + (1.0 - probability) * 20.0);
        if (!near(outputs[0][0], expected))
        {
            return fail("no-cache numerical mismatch", 3);
        }
    }

    // Case 2: kv_cache=1, past_len>0, current_len>0, multiple KV heads.
    // Validates out_cache_k/v are exact [past, current] concatenations and that
    // attention matches a no-cache reference over the concatenated tensors.
    {
        constexpr int head_dim = 4;
        constexpr int value_dim = 4;
        constexpr int query_len = 2;
        constexpr int past_len = 3;
        constexpr int current_len = 2;
        constexpr int num_q_heads = 4;
        constexpr int num_kv_heads = 2; // GQA-style: 2 groups
        constexpr float scale = 0.5f;

        ncnn::Mat query(head_dim, query_len, num_q_heads);
        ncnn::Mat current_key(head_dim, current_len, num_kv_heads);
        ncnn::Mat current_value(value_dim, current_len, num_kv_heads);
        ncnn::Mat past_key(head_dim, past_len, num_kv_heads);
        ncnn::Mat past_value(value_dim, past_len, num_kv_heads);
        fill_channel_rows(query, 1.0f);
        fill_channel_rows(current_key, 2.0f);
        fill_channel_rows(current_value, 3.0f);
        fill_channel_rows(past_key, 4.0f);
        fill_channel_rows(past_value, 5.0f);

        // Explicit expected concat: rows [0..past_len) = past, [past_len..) = current.
        ncnn::Mat expected_key(head_dim, past_len + current_len, num_kv_heads);
        ncnn::Mat expected_value(value_dim, past_len + current_len, num_kv_heads);
        for (int head = 0; head < num_kv_heads; ++head)
        {
            std::memcpy(expected_key.channel(head).row(0),
                        past_key.channel(head),
                        static_cast<size_t>(head_dim) * past_len * sizeof(float));
            std::memcpy(expected_key.channel(head).row(past_len),
                        current_key.channel(head),
                        static_cast<size_t>(head_dim) * current_len * sizeof(float));
            std::memcpy(expected_value.channel(head).row(0),
                        past_value.channel(head),
                        static_cast<size_t>(value_dim) * past_len * sizeof(float));
            std::memcpy(expected_value.channel(head).row(past_len),
                        current_value.channel(head),
                        static_cast<size_t>(value_dim) * current_len * sizeof(float));
        }

        ncnn::Mat reference_output;
        if (reference_attention(query, expected_key, expected_value, scale, &reference_output) != 0)
        {
            return fail("reference attention failed", 4);
        }

        std::unique_ptr<ncnn::Layer> layer(hunyuan_ocr::create_precise_sdpa_layer(nullptr));
        ncnn::ParamDict params;
        params.set(5, 0); // no mask
        params.set(6, scale);
        params.set(7, 1); // kv_cache enabled
        if (layer->load_param(params) != 0)
        {
            return fail("load_param kv-cache failed", 5);
        }

        // bottoms: query, current_key, current_value, past_key, past_value
        std::vector<ncnn::Mat> bottoms = {
            query, current_key, current_value, past_key, past_value};
        std::vector<ncnn::Mat> tops(3); // out, out_cache_k, out_cache_v
        ncnn::Option options;
        options.num_threads = 4; // exercise multi-thread concat path when present
        if (layer->forward(bottoms, tops, options) != 0)
        {
            return fail("kv-cache forward failed", 6);
        }

        if (tops[0].empty() || tops[1].empty() || tops[2].empty())
        {
            return fail("kv-cache tops missing", 7);
        }
        if (tops[1].w != head_dim || tops[1].h != past_len + current_len ||
            tops[1].c != num_kv_heads || tops[2].w != value_dim ||
            tops[2].h != past_len + current_len || tops[2].c != num_kv_heads)
        {
            return fail("out_cache shape mismatch", 8);
        }

        // Exact byte/element equality for concatenated caches.
        if (!mats_equal(tops[1], expected_key, 0.0f))
        {
            return fail("out_cache_k is not exact [past,current] concat", 9);
        }
        if (!mats_equal(tops[2], expected_value, 0.0f))
        {
            return fail("out_cache_v is not exact [past,current] concat", 10);
        }
        if (!mats_equal(tops[0], reference_output, 1e-5f))
        {
            return fail("kv-cache attention output differs from reference", 11);
        }

        // Also run with single thread to ensure sequential path matches.
        std::vector<ncnn::Mat> tops_st(3);
        options.num_threads = 1;
        if (layer->forward(bottoms, tops_st, options) != 0)
        {
            return fail("kv-cache single-thread forward failed", 12);
        }
        if (!mats_equal(tops_st[1], expected_key, 0.0f) ||
            !mats_equal(tops_st[2], expected_value, 0.0f) ||
            !mats_equal(tops_st[0], reference_output, 1e-5f))
        {
            return fail("single-thread kv-cache path mismatch", 13);
        }
    }

    return 0;
}
