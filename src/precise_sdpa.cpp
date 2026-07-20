#include "hunyuan_ocr/precise_sdpa.h"

#include "kv_cache_capacity.h"
#include "precise_sdpa_policy.h"
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
#include "precise_sdpa_profile.h"
#endif

#include <algorithm>
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
#include <chrono>
#endif
#include <cmath>
#include <cstring>
#include <limits>
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
#include <mutex>
#endif
#include <memory>
#include <string>
#include <vector>

namespace hunyuan_ocr {
namespace {

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)

using Clock = std::chrono::steady_clock;

struct ThreadLocalSdpaProfile {
    std::uint64_t call_count = 0;
    double total_ms = 0.0;
    double kv_concat_ms = 0.0;
    double kv_alloc_ms = 0.0;
    double kv_copy_ms = 0.0;
    double attention_compute_ms = 0.0;
    double output_alloc_ms = 0.0;
    std::uint64_t past_kv_copy_bytes = 0;
    std::uint64_t current_kv_copy_bytes = 0;
    int last_query_len = 0;
    int last_key_len = 0;
    int last_num_heads = 0;
    int last_num_threads = 0;
};

ThreadLocalSdpaProfile& tls_profile()
{
    thread_local ThreadLocalSdpaProfile local;
    return local;
}

std::mutex g_profile_mutex;
detail::PreciseSdpaProfileSnapshot g_profile;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void flush_thread_local_profile()
{
    ThreadLocalSdpaProfile& local = tls_profile();
    if (local.call_count == 0 &&
        local.attention_compute_ms == 0.0 &&
        local.kv_concat_ms == 0.0 &&
        local.kv_alloc_ms == 0.0 &&
        local.kv_copy_ms == 0.0 &&
        local.output_alloc_ms == 0.0 &&
        local.total_ms == 0.0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_profile.call_count += local.call_count;
    g_profile.total_ms += local.total_ms;
    g_profile.kv_concat_ms += local.kv_concat_ms;
    g_profile.kv_alloc_ms += local.kv_alloc_ms;
    g_profile.kv_copy_ms += local.kv_copy_ms;
    g_profile.attention_compute_ms += local.attention_compute_ms;
    g_profile.output_alloc_ms += local.output_alloc_ms;
    g_profile.past_kv_copy_bytes += local.past_kv_copy_bytes;
    g_profile.current_kv_copy_bytes += local.current_kv_copy_bytes;
    if (local.last_query_len > 0)
    {
        g_profile.last_query_len = local.last_query_len;
        g_profile.last_key_len = local.last_key_len;
        g_profile.last_num_heads = local.last_num_heads;
        g_profile.last_num_threads = local.last_num_threads;
    }
    local = ThreadLocalSdpaProfile{};
}

#endif

class PreciseSDPA final : public ncnn::Layer {
public:
    PreciseSDPA()
    {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
#if NCNN_VULKAN
        support_vulkan = true;
        support_vulkan_packing = false;
        support_vulkan_any_packing = false;
#endif
    }

    int load_param(const ncnn::ParamDict& pd) override
    {
        attn_mask_ = pd.get(5, 0);
        scale_ = pd.get(6, 0.0f);
        kv_cache_ = pd.get(7, 0);
        return 0;
    }

    int create_pipeline(const ncnn::Option& opt) override
    {
        native_sdpa_.reset(ncnn::create_layer_cpu("SDPA"));
        if (!native_sdpa_)
        {
            return 0;
        }

        ncnn::ParamDict pd;
        pd.set(5, attn_mask_);
        pd.set(6, scale_);
        pd.set(7, kv_cache_);
        if (native_sdpa_->load_param(pd) != 0)
        {
            native_sdpa_.reset();
            return 0;
        }
        if (native_sdpa_->create_pipeline(opt) != 0)
        {
            native_sdpa_->destroy_pipeline(opt);
            native_sdpa_.reset();
        }
#if NCNN_VULKAN
        native_sdpa_vulkan_.reset();
        if (opt.use_vulkan_compute)
        {
            native_sdpa_vulkan_.reset(ncnn::create_layer_vulkan("SDPA"));
            if (!native_sdpa_vulkan_)
            {
                return -1;
            }
            native_sdpa_vulkan_->vkdev = vkdev;
            if (native_sdpa_vulkan_->load_param(pd) != 0 ||
                native_sdpa_vulkan_->create_pipeline(opt) != 0)
            {
                native_sdpa_vulkan_->destroy_pipeline(opt);
                native_sdpa_vulkan_.reset();
                return -1;
            }
        }
#endif
        return 0;
    }

    int destroy_pipeline(const ncnn::Option& opt) override
    {
        int result = 0;
        if (native_sdpa_)
        {
            result = native_sdpa_->destroy_pipeline(opt);
        }
        native_sdpa_.reset();
#if NCNN_VULKAN
        if (native_sdpa_vulkan_)
        {
            const int vulkan_result = native_sdpa_vulkan_->destroy_pipeline(opt);
            if (result == 0) result = vulkan_result;
        }
        native_sdpa_vulkan_.reset();
#endif
        return result;
    }

#if NCNN_VULKAN
    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs,
                ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override
    {
        if (!native_sdpa_vulkan_)
        {
            return -1;
        }
        return native_sdpa_vulkan_->forward(bottom_blobs, top_blobs, cmd, opt);
    }
#endif

    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override
    {
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
        const auto total_start = Clock::now();
        ThreadLocalSdpaProfile& local = tls_profile();
#endif
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

        if (native_sdpa_ && detail::should_use_native_sdpa_prefill(kv_cache_ != 0,
                                                                   past_len,
                                                                   query_len))
        {
            return native_sdpa_->forward(bottom_blobs, top_blobs, opt);
        }

        ncnn::Mat key = current_key;
        ncnn::Mat value = current_value;
        if (past_len > 0)
        {
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
            detail::KvCacheAppendProfile append_profile;
#endif
            std::string kv_error;
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
            const bool appended = detail::append_or_grow_kv_cache_profiled(past_key,
                                                                           past_value,
                                                                           current_key,
                                                                           current_value,
                                                                           &key,
                                                                           &value,
                                                                           &append_profile,
                                                                           &kv_error,
                                                                           opt.blob_allocator);
#else
            const bool appended = detail::append_or_grow_kv_cache(past_key,
                                                                  past_value,
                                                                  current_key,
                                                                  current_value,
                                                                  &key,
                                                                  &value,
                                                                  &kv_error,
                                                                  opt.blob_allocator);
#endif
            if (!appended)
            {
                return -100;
            }
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
            local.kv_alloc_ms += append_profile.allocation_ms;
            local.kv_copy_ms += append_profile.copy_ms;
            local.kv_concat_ms += append_profile.allocation_ms + append_profile.copy_ms;
            local.past_kv_copy_bytes += append_profile.past_copy_bytes;
            local.current_kv_copy_bytes += append_profile.current_copy_bytes;
#endif
        }

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
        const auto output_alloc_start = Clock::now();
#endif
        ncnn::Mat& output = top_blobs[0];
        output.create(value_dim, query_len, num_heads, 4u, opt.blob_allocator);
        if (output.empty()) return -100;
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
        local.output_alloc_ms += elapsed_ms(output_alloc_start, Clock::now());
        // Coarse attention timer: one wall sample around the whole parallel
        // head loop. No per-head/per-row timestamps (keeps profile overhead low).
        const auto attention_start = Clock::now();
#endif

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
                        sum += probabilities[static_cast<size_t>(key_index)] *
                               value_head.row(key_index)[dim];
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
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
        // Full attention body wall time (QK+softmax+PV combined under coarse mode).
        local.attention_compute_ms += elapsed_ms(attention_start, Clock::now());
        local.call_count += 1;
        local.total_ms += elapsed_ms(total_start, Clock::now());
        local.last_query_len = query_len;
        local.last_key_len = key_len;
        local.last_num_heads = num_heads;
        local.last_num_threads = opt.num_threads;
        // Do not mutex-flush here: microbench is single-caller and accumulates
        // in TLS until reset/snapshot. Avoids fixed per-forward lock overhead.
#endif
        return 0;
    }

private:
    int attn_mask_ = 0;
    float scale_ = 0.0f;
    int kv_cache_ = 0;
    std::unique_ptr<ncnn::Layer> native_sdpa_;
#if NCNN_VULKAN
    std::unique_ptr<ncnn::Layer> native_sdpa_vulkan_;
#endif
};

} // namespace

ncnn::Layer* create_precise_sdpa_layer(void*)
{
    return new PreciseSDPA;
}

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)

namespace detail {

void reset_precise_sdpa_profile()
{
    flush_thread_local_profile();
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_profile = PreciseSdpaProfileSnapshot{};
}

PreciseSdpaProfileSnapshot snapshot_precise_sdpa_profile()
{
    flush_thread_local_profile();
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    PreciseSdpaProfileSnapshot snapshot = g_profile;
    // Prefer explicit alloc+copy sum when both were recorded.
    if (snapshot.kv_alloc_ms > 0.0 || snapshot.kv_copy_ms > 0.0)
    {
        snapshot.kv_concat_ms = snapshot.kv_alloc_ms + snapshot.kv_copy_ms;
    }
    const double accounted = snapshot.kv_concat_ms + snapshot.attention_compute_ms +
                             snapshot.output_alloc_ms;
    snapshot.other_ms = std::max(0.0, snapshot.total_ms - accounted);
    return snapshot;
}

} // namespace detail

#endif

} // namespace hunyuan_ocr
