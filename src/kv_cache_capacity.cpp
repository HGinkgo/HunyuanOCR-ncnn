#include "kv_cache_capacity.h"

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
#include <chrono>
#endif
#include <cstring>

namespace hunyuan_ocr {
namespace detail {

bool make_kv_cache_logical_view(const ncnn::Mat& past,
                                int logical_rows,
                                ncnn::Mat* output,
                                std::string* error)
{
    if (output == nullptr)
    {
        if (error) *error = "kv logical view output is null";
        return false;
    }
    const int capacity = kv_cache_capacity_rows(past);
    if (capacity <= 0 || logical_rows < 0 || logical_rows > capacity)
    {
        if (error) *error = "kv logical view requires capacity-bearing past";
        return false;
    }
    // Shallow copy: keep data/refcount/cstep/nstep; only h is the logical length.
    ncnn::Mat view = past;
    view.h = logical_rows;
    *output = std::move(view);
    return true;
}

bool allocate_kv_cache_with_capacity(int width,
                                     int logical_rows,
                                     int capacity_rows,
                                     int channels,
                                     ncnn::Mat* output,
                                     std::string* error,
                                     ncnn::Allocator* allocator)
{
    if (output == nullptr ||
        width <= 0 ||
        channels <= 0 ||
        logical_rows < 0 ||
        capacity_rows < logical_rows ||
        capacity_rows <= 0)
    {
        if (error) *error = "invalid kv capacity allocation arguments";
        return false;
    }
    // Create with h=capacity so cstep covers the reserved rows.
    ncnn::Mat backing(width, capacity_rows, channels, 4u, allocator);
    if (backing.empty())
    {
        if (error) *error = "kv capacity allocation failed";
        return false;
    }
    backing.h = logical_rows;
    *output = std::move(backing);
    return true;
}

bool copy_kv_cache_rows(const ncnn::Mat& source,
                        int source_row,
                        int rows,
                        ncnn::Mat* dest,
                        int dest_row,
                        std::string* error)
{
    if (dest == nullptr ||
        source.empty() ||
        dest->empty() ||
        source.dims != 3 ||
        dest->dims != 3 ||
        source.elemsize != 4u ||
        dest->elemsize != 4u ||
        source.elempack != 1 ||
        dest->elempack != 1 ||
        source.w <= 0 ||
        source.w != dest->w ||
        source.c != dest->c ||
        rows < 0 ||
        source_row < 0 ||
        dest_row < 0)
    {
        if (error) *error = "invalid kv row copy";
        return false;
    }
    if (rows == 0)
    {
        return true;
    }
    const int capacity = kv_cache_capacity_rows(*dest);
    const int dest_limit = capacity > 0 ? capacity : dest->h;
    if (source_row > source.h || rows > source.h - source_row ||
        dest_row > dest_limit || rows > dest_limit - dest_row)
    {
        if (error) *error = "kv row copy out of range";
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(source.w) * sizeof(float);
    if (static_cast<size_t>(rows) > std::numeric_limits<size_t>::max() / row_bytes)
    {
        if (error) *error = "kv row copy size overflow";
        return false;
    }
    const size_t bytes = row_bytes * static_cast<size_t>(rows);
    for (int channel = 0; channel < source.c; ++channel)
    {
        std::memcpy(dest->channel(channel).row(dest_row),
                    source.channel(channel).row(source_row),
                    bytes);
    }
    return true;
}

namespace {

struct KvCacheAppendPlan {
    int past_len = 0;
    int current_len = 0;
    int key_len = 0;
    int capacity_rows = 0;
    bool inplace = false;
};

bool build_append_plan(const ncnn::Mat& past_key,
                       const ncnn::Mat& past_value,
                       const ncnn::Mat& current_key,
                       const ncnn::Mat& current_value,
                       KvCacheAppendPlan* plan,
                       std::string* error)
{
    if (plan == nullptr)
    {
        if (error) *error = "kv append plan is null";
        return false;
    }
    if (past_key.dims != 3 || past_value.dims != 3 ||
        current_key.dims != 3 || current_value.dims != 3 ||
        past_key.elemsize != 4u || past_value.elemsize != 4u ||
        current_key.elemsize != 4u || current_value.elemsize != 4u ||
        past_key.elempack != 1 || past_value.elempack != 1 ||
        current_key.elempack != 1 || current_value.elempack != 1 ||
        past_key.w != current_key.w || past_value.w != current_value.w ||
        past_key.c != current_key.c || past_value.c != current_value.c ||
        past_key.c != past_value.c ||
        past_key.h != past_value.h || current_key.h != current_value.h ||
        past_key.h < 0 || current_key.h <= 0)
    {
        if (error) *error = "kv append input shape mismatch";
        return false;
    }

    if (current_key.h > std::numeric_limits<int>::max() - past_key.h)
    {
        if (error) *error = "kv append length overflow";
        return false;
    }

    plan->past_len = past_key.h;
    plan->current_len = current_key.h;
    plan->key_len = plan->past_len + plan->current_len;
    const int key_capacity = kv_cache_capacity_rows(past_key);
    const int value_capacity = kv_cache_capacity_rows(past_value);

    plan->inplace =
        key_capacity > 0 &&
        value_capacity == key_capacity &&
        plan->past_len <= key_capacity &&
        plan->key_len <= key_capacity;
    plan->capacity_rows = plan->inplace
        ? key_capacity
        : round_up_kv_capacity_rows(plan->key_len);
    if (plan->capacity_rows <= 0)
    {
        if (error) *error = "kv append capacity overflow";
        return false;
    }
    return true;
}

bool allocate_append_outputs(const ncnn::Mat& past_key,
                             const ncnn::Mat& past_value,
                             const KvCacheAppendPlan& plan,
                             ncnn::Mat* key,
                             ncnn::Mat* value,
                             std::string* error,
                             ncnn::Allocator* allocator)
{
    return allocate_kv_cache_with_capacity(past_key.w,
                                           plan.key_len,
                                           plan.capacity_rows,
                                           past_key.c,
                                           key,
                                           error,
                                           allocator) &&
           allocate_kv_cache_with_capacity(past_value.w,
                                           plan.key_len,
                                           plan.capacity_rows,
                                           past_value.c,
                                           value,
                                           error,
                                           allocator);
}

bool copy_append_rows(const ncnn::Mat& past_key,
                      const ncnn::Mat& past_value,
                      const ncnn::Mat& current_key,
                      const ncnn::Mat& current_value,
                      const KvCacheAppendPlan& plan,
                      ncnn::Mat* key,
                      ncnn::Mat* value,
                      std::string* error)
{
    if (!plan.inplace && plan.past_len > 0)
    {
        if (!copy_kv_cache_rows(past_key, 0, plan.past_len, key, 0, error) ||
            !copy_kv_cache_rows(past_value, 0, plan.past_len, value, 0, error))
        {
            return false;
        }
    }
    return copy_kv_cache_rows(current_key, 0, plan.current_len, key, plan.past_len, error) &&
           copy_kv_cache_rows(current_value, 0, plan.current_len, value, plan.past_len, error);
}

} // namespace


bool append_or_grow_kv_cache(const ncnn::Mat& past_key,
                             const ncnn::Mat& past_value,
                             const ncnn::Mat& current_key,
                             const ncnn::Mat& current_value,
                             ncnn::Mat* key,
                             ncnn::Mat* value,
                             std::string* error,
                             ncnn::Allocator* allocator)
{
    if (key == nullptr || value == nullptr)
    {
        if (error) *error = "kv append output is null";
        return false;
    }

    KvCacheAppendPlan plan;
    if (!build_append_plan(past_key, past_value, current_key, current_value, &plan, error))
    {
        return false;
    }

    if (plan.inplace)
    {
        if (!make_kv_cache_logical_view(past_key, plan.key_len, key, error) ||
            !make_kv_cache_logical_view(past_value, plan.key_len, value, error))
        {
            return false;
        }
    }
    else if (!allocate_append_outputs(past_key,
                                      past_value,
                                      plan,
                                      key,
                                      value,
                                      error,
                                      allocator))
    {
        return false;
    }

    return copy_append_rows(past_key,
                            past_value,
                            current_key,
                            current_value,
                            plan,
                            key,
                            value,
                            error);
}

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)

bool append_or_grow_kv_cache_profiled(const ncnn::Mat& past_key,
                                      const ncnn::Mat& past_value,
                                      const ncnn::Mat& current_key,
                                      const ncnn::Mat& current_value,
                                      ncnn::Mat* key,
                                      ncnn::Mat* value,
                                      KvCacheAppendProfile* profile,
                                      std::string* error,
                                      ncnn::Allocator* allocator)
{
    if (key == nullptr || value == nullptr || profile == nullptr)
    {
        if (error) *error = "profiled kv append output is null";
        return false;
    }

    KvCacheAppendPlan plan;
    if (!build_append_plan(past_key, past_value, current_key, current_value, &plan, error))
    {
        return false;
    }

    *profile = KvCacheAppendProfile{};
    profile->past_copy_bytes = plan.inplace
        ? 0
        : static_cast<std::uint64_t>(past_key.c) *
              static_cast<std::uint64_t>(plan.past_len) *
              static_cast<std::uint64_t>(past_key.w + past_value.w) * sizeof(float);
    profile->current_copy_bytes =
        static_cast<std::uint64_t>(current_key.c) *
        static_cast<std::uint64_t>(plan.current_len) *
        static_cast<std::uint64_t>(current_key.w + current_value.w) * sizeof(float);

    if (plan.inplace)
    {
        if (!make_kv_cache_logical_view(past_key, plan.key_len, key, error) ||
            !make_kv_cache_logical_view(past_value, plan.key_len, value, error))
        {
            return false;
        }
    }
    else
    {
        const auto allocation_start = std::chrono::steady_clock::now();
        const bool allocated = allocate_append_outputs(past_key,
                                                       past_value,
                                                       plan,
                                                       key,
                                                       value,
                                                       error,
                                                       allocator);
        profile->allocation_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - allocation_start)
                                     .count();
        if (!allocated)
        {
            return false;
        }
    }

    const auto copy_start = std::chrono::steady_clock::now();
    const bool copied = copy_append_rows(past_key,
                                         past_value,
                                         current_key,
                                         current_value,
                                         plan,
                                         key,
                                         value,
                                         error);
    profile->copy_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - copy_start)
                           .count();
    return copied;
}

#endif

} // namespace detail
} // namespace hunyuan_ocr
