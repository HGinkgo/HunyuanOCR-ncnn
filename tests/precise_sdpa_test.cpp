#include "hunyuan_ocr/precise_sdpa.h"

#include "kv_cache_capacity.h"
#include "precise_sdpa_policy.h"

#include <allocator.h>

#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

bool nearly_equal(float value, float expected, float tol = 1e-5f)
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

bool mats_equal_logical(const ncnn::Mat& left, const ncnn::Mat& right, float tol = 0.0f)
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

int reference_attention(const ncnn::Mat& query,
                        const ncnn::Mat& key,
                        const ncnn::Mat& value,
                        float scale,
                        ncnn::Mat* output)
{
    std::unique_ptr<ncnn::Layer> layer(hunyuan_ocr::create_precise_sdpa_layer(nullptr));
    ncnn::ParamDict params;
    params.set(5, 0);
    params.set(6, scale);
    params.set(7, 0);
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

void build_expected_concat(const ncnn::Mat& past,
                           const ncnn::Mat& current,
                           ncnn::Mat* expected)
{
    expected->create(past.w, past.h + current.h, past.c);
    for (int head = 0; head < past.c; ++head)
    {
        std::memcpy(expected->channel(head).row(0),
                    past.channel(head),
                    static_cast<size_t>(past.w) * past.h * sizeof(float));
        std::memcpy(expected->channel(head).row(past.h),
                    current.channel(head),
                    static_cast<size_t>(current.w) * current.h * sizeof(float));
    }
}

bool run_kv_forward(const ncnn::Mat& query,
                    const ncnn::Mat& current_key,
                    const ncnn::Mat& current_value,
                    const ncnn::Mat& past_key,
                    const ncnn::Mat& past_value,
                    float scale,
                    int num_threads,
                    std::vector<ncnn::Mat>* tops,
                    ncnn::Allocator* blob_allocator = nullptr)
{
    std::unique_ptr<ncnn::Layer> layer(hunyuan_ocr::create_precise_sdpa_layer(nullptr));
    ncnn::ParamDict params;
    params.set(5, 0);
    params.set(6, scale);
    params.set(7, 1);
    if (layer->load_param(params) != 0)
    {
        return false;
    }
    tops->assign(3, ncnn::Mat());
    ncnn::Option options;
    options.num_threads = num_threads;
    options.blob_allocator = blob_allocator;
    return layer->forward(
               {query, current_key, current_value, past_key, past_value}, *tops, options) == 0;
}

} // namespace

int main()
{
    using hunyuan_ocr::detail::allocate_kv_cache_with_capacity;
    using hunyuan_ocr::detail::append_or_grow_kv_cache;
    using hunyuan_ocr::detail::kKvCacheCapacityChunkRows;
    using hunyuan_ocr::detail::kv_cache_capacity_rows;
    using hunyuan_ocr::detail::make_kv_cache_logical_view;
    using hunyuan_ocr::detail::round_up_kv_capacity_rows;
    using hunyuan_ocr::detail::should_use_native_sdpa_prefill;

    if (!should_use_native_sdpa_prefill(true, 0, 533) ||
        should_use_native_sdpa_prefill(false, 0, 533) ||
        should_use_native_sdpa_prefill(true, 533, 1) ||
        should_use_native_sdpa_prefill(true, 533, 16))
    {
        return fail("hybrid SDPA routing policy mismatch", 34);
    }

    // Exercise the same built-in CPU SDPA pipeline used by cache-enabled prefill.
    {
        std::unique_ptr<ncnn::Layer> hybrid(hunyuan_ocr::create_precise_sdpa_layer(nullptr));
        std::unique_ptr<ncnn::Layer> native(ncnn::create_layer_cpu("SDPA"));
        if (!native)
        {
            return fail("native SDPA layer is unavailable", 35);
        }

        ncnn::ParamDict params;
        params.set(5, 0);
        params.set(6, 0.5f);
        params.set(7, 1);
        ncnn::Option options;
        options.num_threads = 2;
        if (hybrid->load_param(params) != 0 || native->load_param(params) != 0 ||
            hybrid->create_pipeline(options) != 0 || native->create_pipeline(options) != 0)
        {
            return fail("hybrid prefill pipeline creation failed", 36);
        }

        ncnn::Mat query(4, 2, 2);
        ncnn::Mat key(4, 2, 1);
        ncnn::Mat value(4, 2, 1);
        fill_channel_rows(query, 0.3f);
        fill_channel_rows(key, 0.4f);
        fill_channel_rows(value, 0.5f);
        const std::vector<ncnn::Mat> bottoms = {query, key, value, ncnn::Mat(), ncnn::Mat()};
        std::vector<ncnn::Mat> hybrid_outputs(3);
        std::vector<ncnn::Mat> native_outputs(3);
        if (hybrid->forward(bottoms, hybrid_outputs, options) != 0 ||
            native->forward(bottoms, native_outputs, options) != 0)
        {
            return fail("hybrid prefill forward failed", 37);
        }
        for (size_t index = 0; index < hybrid_outputs.size(); ++index)
        {
            if (!mats_equal_logical(hybrid_outputs[index], native_outputs[index], 0.0f))
            {
                return fail("hybrid prefill did not use native SDPA", 38);
            }
        }
        if (hybrid->destroy_pipeline(options) != 0 || native->destroy_pipeline(options) != 0)
        {
            return fail("hybrid prefill pipeline destruction failed", 39);
        }
    }

    if (round_up_kv_capacity_rows(INT_MAX) != 0)
    {
        return fail("overflowing KV capacity must be rejected", 31);
    }

    {
        ncnn::UnlockedPoolAllocator allocator;
        ncnn::Mat allocated;
        std::string error;
        if (!allocate_kv_cache_with_capacity(8, 4, 16, 2, &allocated, &error, &allocator) ||
            allocated.allocator != &allocator)
        {
            return fail("KV capacity allocation must preserve the requested allocator", 32);
        }
        allocated.release();
        allocator.clear();
    }

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
        if (!nearly_equal(outputs[0][0], expected))
        {
            return fail("no-cache numerical mismatch", 3);
        }
    }

    // Case 2: exact-size past (no spare capacity) must still produce exact concat.
    {
        constexpr int head_dim = 4;
        constexpr int value_dim = 4;
        constexpr int query_len = 2;
        constexpr int past_len = 3;
        constexpr int current_len = 2;
        constexpr int num_q_heads = 4;
        constexpr int num_kv_heads = 2;
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

        // Exact-size past: capacity_rows == h (or no spare room for append).
        if (kv_cache_capacity_rows(past_key) < past_len)
        {
            return fail("exact-size past capacity probe failed", 4);
        }

        ncnn::Mat expected_key;
        ncnn::Mat expected_value;
        build_expected_concat(past_key, current_key, &expected_key);
        build_expected_concat(past_value, current_value, &expected_value);

        ncnn::Mat reference_output;
        if (reference_attention(query, expected_key, expected_value, scale, &reference_output) != 0)
        {
            return fail("reference attention failed", 5);
        }

        ncnn::UnlockedPoolAllocator allocator;
        for (int threads : {1, 4})
        {
            std::vector<ncnn::Mat> tops;
            if (!run_kv_forward(query,
                                current_key,
                                current_value,
                                past_key,
                                past_value,
                                scale,
                                threads,
                                &tops,
                                &allocator))
            {
                return fail("exact-size kv-cache forward failed", 6);
            }
            if (!mats_equal_logical(tops[1], expected_key, 0.0f) ||
                !mats_equal_logical(tops[2], expected_value, 0.0f) ||
                !mats_equal_logical(tops[0], reference_output, 1e-5f))
            {
                return fail("exact-size fallback concat mismatch", 7);
            }
            if (tops[1].allocator != &allocator || tops[2].allocator != &allocator)
            {
                return fail("PreciseSDPA grow must use opt.blob_allocator", 33);
            }
            // Fallback growth should expose capacity for later appends.
            if (kv_cache_capacity_rows(tops[1]) < past_len + current_len ||
                kv_cache_capacity_rows(tops[1]) % kKvCacheCapacityChunkRows != 0)
            {
                return fail("fallback out_cache lacks rounded capacity", 8);
            }
        }
        allocator.clear();
    }

    // Case 3: capacity-bearing past appends inplace (data pointer / cstep stable).
    {
        constexpr int head_dim = 8;
        constexpr int past_len = 5;
        constexpr int current_len = 3;
        constexpr int capacity = 16;
        constexpr int num_kv_heads = 2;
        std::string error;
        ncnn::Mat past_key;
        ncnn::Mat past_value;
        if (!allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_key, &error) ||
            !allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_value, &error))
        {
            return fail(error.c_str(), 9);
        }
        fill_channel_rows(past_key, 1.0f);
        fill_channel_rows(past_value, 2.0f);
        ncnn::Mat current_key(head_dim, current_len, num_kv_heads);
        ncnn::Mat current_value(head_dim, current_len, num_kv_heads);
        fill_channel_rows(current_key, 3.0f);
        fill_channel_rows(current_value, 4.0f);

        void* past_key_data = past_key.data;
        void* past_value_data = past_value.data;
        const size_t past_key_cstep = past_key.cstep;
        const size_t past_value_cstep = past_value.cstep;

        ncnn::Mat out_key;
        ncnn::Mat out_value;
        if (!append_or_grow_kv_cache(past_key,
                                     past_value,
                                     current_key,
                                     current_value,
                                     &out_key,
                                     &out_value,
                                     &error))
        {
            return fail(error.c_str(), 10);
        }
        if (out_key.data != past_key_data || out_value.data != past_value_data ||
            out_key.cstep != past_key_cstep || out_value.cstep != past_value_cstep ||
            out_key.h != past_len + current_len || out_value.h != past_len + current_len)
        {
            return fail("inplace append did not preserve backing data/cstep/h", 11);
        }

        ncnn::Mat packed_past_k(head_dim, past_len, num_kv_heads);
        ncnn::Mat packed_past_v(head_dim, past_len, num_kv_heads);
        for (int c = 0; c < num_kv_heads; ++c)
        {
            std::memcpy(packed_past_k.channel(c),
                        past_key.channel(c),
                        static_cast<size_t>(head_dim) * past_len * sizeof(float));
            std::memcpy(packed_past_v.channel(c),
                        past_value.channel(c),
                        static_cast<size_t>(head_dim) * past_len * sizeof(float));
        }
        ncnn::Mat expected_key;
        ncnn::Mat expected_value;
        build_expected_concat(packed_past_k, current_key, &expected_key);
        build_expected_concat(packed_past_v, current_value, &expected_value);
        if (!mats_equal_logical(out_key, expected_key, 0.0f) ||
            !mats_equal_logical(out_value, expected_value, 0.0f))
        {
            return fail("inplace append content mismatch", 12);
        }
    }

    // Case 4: speculative append 16 rows, shrink logical h, re-append overwrites reject zone.
    {
        constexpr int head_dim = 4;
        constexpr int past_len = 8;
        constexpr int block = 16;
        constexpr int accept = 5;
        constexpr int capacity = 64;
        constexpr int num_kv_heads = 2;
        std::string error;
        ncnn::Mat past_key;
        ncnn::Mat past_value;
        if (!allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_key, &error) ||
            !allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_value, &error))
        {
            return fail(error.c_str(), 13);
        }
        fill_channel_rows(past_key, 0.5f);
        fill_channel_rows(past_value, 0.6f);

        ncnn::Mat draft_key(head_dim, block, num_kv_heads);
        ncnn::Mat draft_value(head_dim, block, num_kv_heads);
        fill_channel_rows(draft_key, 7.0f);
        fill_channel_rows(draft_value, 8.0f);

        ncnn::Mat key;
        ncnn::Mat value;
        if (!append_or_grow_kv_cache(past_key, past_value, draft_key, draft_value, &key, &value, &error))
        {
            return fail(error.c_str(), 14);
        }
        if (key.h != past_len + block)
        {
            return fail("speculative append length mismatch", 15);
        }
        // Shrink to accepted prefix (like DFlash view_dflash_rows).
        ncnn::Mat committed_key;
        ncnn::Mat committed_value;
        if (!make_kv_cache_logical_view(key, past_len + accept, &committed_key, &error) ||
            !make_kv_cache_logical_view(value, past_len + accept, &committed_value, &error))
        {
            return fail(error.c_str(), 16);
        }
        if (committed_key.data != key.data || committed_key.cstep != key.cstep ||
            committed_key.h != past_len + accept)
        {
            return fail("shrink view lost capacity backing", 17);
        }

        ncnn::Mat next_key(head_dim, 2, num_kv_heads);
        ncnn::Mat next_value(head_dim, 2, num_kv_heads);
        fill_channel_rows(next_key, 9.0f);
        fill_channel_rows(next_value, 10.0f);
        ncnn::Mat out_key;
        ncnn::Mat out_value;
        if (!append_or_grow_kv_cache(committed_key,
                                     committed_value,
                                     next_key,
                                     next_value,
                                     &out_key,
                                     &out_value,
                                     &error))
        {
            return fail(error.c_str(), 18);
        }
        if (out_key.data != key.data || out_key.h != past_len + accept + 2)
        {
            return fail("re-append after shrink did not reuse capacity", 19);
        }
        // Rejected draft rows must be overwritten by next append.
        for (int c = 0; c < num_kv_heads; ++c)
        {
            for (int r = 0; r < 2; ++r)
            {
                for (int w = 0; w < head_dim; ++w)
                {
                    if (out_key.channel(c).row(past_len + accept + r)[w] !=
                        next_key.channel(c).row(r)[w])
                    {
                        return fail("reject zone not overwritten on re-append", 20);
                    }
                }
            }
        }
    }

    // Case 5: capacity exhaustion grows with exact content.
    {
        constexpr int head_dim = 4;
        constexpr int past_len = 6;
        constexpr int capacity = 8;
        constexpr int current_len = 4; // past+current > capacity
        constexpr int num_kv_heads = 2;
        std::string error;
        ncnn::Mat past_key;
        ncnn::Mat past_value;
        if (!allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_key, &error) ||
            !allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_value, &error))
        {
            return fail(error.c_str(), 21);
        }
        fill_channel_rows(past_key, 1.1f);
        fill_channel_rows(past_value, 1.2f);
        ncnn::Mat current_key(head_dim, current_len, num_kv_heads);
        ncnn::Mat current_value(head_dim, current_len, num_kv_heads);
        fill_channel_rows(current_key, 2.1f);
        fill_channel_rows(current_value, 2.2f);

        ncnn::Mat out_key;
        ncnn::Mat out_value;
        if (!append_or_grow_kv_cache(past_key,
                                     past_value,
                                     current_key,
                                     current_value,
                                     &out_key,
                                     &out_value,
                                     &error))
        {
            return fail(error.c_str(), 22);
        }
        if (out_key.data == past_key.data)
        {
            return fail("capacity exhaustion should reallocate", 23);
        }
        if (kv_cache_capacity_rows(out_key) !=
            round_up_kv_capacity_rows(past_len + current_len))
        {
            return fail("grown capacity not rounded to chunk", 24);
        }
        ncnn::Mat packed_past(head_dim, past_len, num_kv_heads);
        for (int c = 0; c < num_kv_heads; ++c)
        {
            std::memcpy(packed_past.channel(c),
                        past_key.channel(c),
                        static_cast<size_t>(head_dim) * past_len * sizeof(float));
        }
        ncnn::Mat expected;
        build_expected_concat(packed_past, current_key, &expected);
        if (!mats_equal_logical(out_key, expected, 0.0f))
        {
            return fail("grown cache content mismatch", 25);
        }
    }

    // Case 6: full SDPA path with capacity past for q=1 and q=16, ST/MT.
    for (int query_len : {1, 16})
    {
        constexpr int head_dim = 4;
        constexpr int past_len = 7;
        constexpr int capacity = 64;
        constexpr int num_q_heads = 4;
        constexpr int num_kv_heads = 2;
        constexpr float scale = 0.25f;
        std::string error;
        ncnn::Mat past_key;
        ncnn::Mat past_value;
        if (!allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_key, &error) ||
            !allocate_kv_cache_with_capacity(head_dim, past_len, capacity, num_kv_heads, &past_value, &error))
        {
            return fail(error.c_str(), 26);
        }
        fill_channel_rows(past_key, 0.2f);
        fill_channel_rows(past_value, 0.3f);
        ncnn::Mat query(head_dim, query_len, num_q_heads);
        ncnn::Mat current_key(head_dim, query_len, num_kv_heads);
        ncnn::Mat current_value(head_dim, query_len, num_kv_heads);
        fill_channel_rows(query, 0.4f);
        fill_channel_rows(current_key, 0.5f);
        fill_channel_rows(current_value, 0.6f);

        ncnn::Mat packed_past_k(head_dim, past_len, num_kv_heads);
        ncnn::Mat packed_past_v(head_dim, past_len, num_kv_heads);
        for (int c = 0; c < num_kv_heads; ++c)
        {
            std::memcpy(packed_past_k.channel(c),
                        past_key.channel(c),
                        static_cast<size_t>(head_dim) * past_len * sizeof(float));
            std::memcpy(packed_past_v.channel(c),
                        past_value.channel(c),
                        static_cast<size_t>(head_dim) * past_len * sizeof(float));
        }
        ncnn::Mat expected_key;
        ncnn::Mat expected_value;
        build_expected_concat(packed_past_k, current_key, &expected_key);
        build_expected_concat(packed_past_v, current_value, &expected_value);
        ncnn::Mat reference_output;
        if (reference_attention(query, expected_key, expected_value, scale, &reference_output) != 0)
        {
            return fail("capacity SDPA reference failed", 27);
        }

        for (int threads : {1, 4})
        {
            std::vector<ncnn::Mat> tops;
            if (!run_kv_forward(query,
                                current_key,
                                current_value,
                                past_key,
                                past_value,
                                scale,
                                threads,
                                &tops))
            {
                return fail("capacity SDPA forward failed", 28);
            }
            if (tops[1].data != past_key.data)
            {
                return fail("capacity SDPA did not append inplace", 29);
            }
            if (!mats_equal_logical(tops[1], expected_key, 0.0f) ||
                !mats_equal_logical(tops[2], expected_value, 0.0f) ||
                !mats_equal_logical(tops[0], reference_output, 1e-5f))
            {
                return fail("capacity SDPA output/content mismatch", 30);
            }
        }
    }

    return 0;
}
