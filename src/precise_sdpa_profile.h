#pragma once

// Internal PreciseSDPA profiling helpers. Only active when
// HUNYUAN_OCR_ENABLE_SDPA_PROFILE is defined at compile time.
// Not part of the public HunyuanOCR-ncnn API.

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace hunyuan_ocr {
namespace detail {

struct PreciseSdpaProfileSnapshot {
    std::uint64_t call_count = 0;
    double total_ms = 0.0;
    // kv_concat_ms remains alloc+copy for backward-compatible reporting.
    double kv_concat_ms = 0.0;
    double kv_alloc_ms = 0.0;
    double kv_copy_ms = 0.0;
    double attention_compute_ms = 0.0;
    double output_alloc_ms = 0.0;
    double other_ms = 0.0;
    std::uint64_t past_kv_copy_bytes = 0;
    std::uint64_t current_kv_copy_bytes = 0;
    int last_query_len = 0;
    int last_key_len = 0;
    int last_num_heads = 0;
    int last_num_threads = 0;
};

inline std::string precise_sdpa_profile_csv_header()
{
    return "context_len,query_len,num_threads,warmup,repeat,decoder_median_ms,"
           "call_count,total_ms,kv_concat_ms,kv_alloc_ms,kv_copy_ms,"
           "attention_compute_ms,"
           "output_alloc_ms,other_ms,past_kv_copy_bytes,current_kv_copy_bytes,"
           "last_query_len,last_key_len,last_num_heads,last_num_threads";
}

inline std::string format_precise_sdpa_profile_csv_row(
    int context_len,
    int query_len,
    int num_threads,
    int warmup,
    int repeat,
    double decoder_median_ms,
    const PreciseSdpaProfileSnapshot& snapshot)
{
    std::ostringstream stream;
    stream << context_len << ','
           << query_len << ','
           << num_threads << ','
           << warmup << ','
           << repeat << ','
           << std::fixed << std::setprecision(3)
           << decoder_median_ms << ','
           << snapshot.call_count << ','
           << snapshot.total_ms << ','
           << snapshot.kv_concat_ms << ','
           << snapshot.kv_alloc_ms << ','
           << snapshot.kv_copy_ms << ','
           << snapshot.attention_compute_ms << ','
           << snapshot.output_alloc_ms << ','
           << snapshot.other_ms << ','
           << snapshot.past_kv_copy_bytes << ','
           << snapshot.current_kv_copy_bytes << ','
           << snapshot.last_query_len << ','
           << snapshot.last_key_len << ','
           << snapshot.last_num_heads << ','
           << snapshot.last_num_threads;
    return stream.str();
}

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)

void reset_precise_sdpa_profile();
PreciseSdpaProfileSnapshot snapshot_precise_sdpa_profile();

#else

inline void reset_precise_sdpa_profile() {}

inline PreciseSdpaProfileSnapshot snapshot_precise_sdpa_profile()
{
    return {};
}

#endif

} // namespace detail
} // namespace hunyuan_ocr
