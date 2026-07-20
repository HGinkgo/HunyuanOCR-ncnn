#pragma once

namespace hunyuan_ocr {
namespace detail {

inline bool should_use_native_sdpa_prefill(bool kv_cache, int past_len, int query_len)
{
    return kv_cache && past_len == 0 && query_len > 1;
}

} // namespace detail
} // namespace hunyuan_ocr
