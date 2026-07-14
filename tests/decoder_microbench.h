#pragma once

#include "text_decoder_step.h"

#include <string>
#include <vector>

#include <net.h>

namespace hunyuan_ocr {
namespace detail {

// exact: past Mat is exact-size (cold first-touch / migration path).
// reserved: past Mat is capacity-bearing (steady no-copy append path).
enum class DecoderMicrobenchCacheMode {
    Exact = 0,
    Reserved = 1,
};

struct DecoderMicrobenchCell {
    int context_len = 0;
    int query_len = 0;
    int num_threads = 0;
    int warmup = 0;
    int repeat = 0;
    DecoderMicrobenchCacheMode cache_mode = DecoderMicrobenchCacheMode::Exact;
};

struct DecoderMicrobenchTiming {
    double min_ms = 0.0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double per_query_token_ms = 0.0;
};

struct DecoderMicrobenchRow {
    DecoderMicrobenchCell cell;
    DecoderMicrobenchTiming timing;
};

// Default P5.2 phase-1 matrix: 3 context x 2 query x 4 threads = 24.
std::vector<DecoderMicrobenchCell> default_decoder_microbench_matrix(int warmup, int repeat);

// Parse comma-separated positive integers, preserving order and dropping empties.
// Rejects non-positive values and partial tokens such as "4x" / "8foo".
// Deduplicates while preserving first occurrence.
bool parse_positive_int_list(const std::string& text,
                             std::vector<int>* values,
                             std::string* error);

DecoderMicrobenchTiming compute_timing_stats(const std::vector<double>& samples_ms,
                                             int query_len);

std::string decoder_microbench_csv_header();
std::string format_decoder_microbench_csv_row(const DecoderMicrobenchRow& row);

// Expand full cartesian product in stable order:
// context_lens x query_lens x thread_counts.
std::vector<DecoderMicrobenchCell> expand_decoder_microbench_matrix(
    const std::vector<int>& context_lens,
    const std::vector<int>& query_lens,
    const std::vector<int>& thread_counts,
    int warmup,
    int repeat,
    std::string* error);

// Stable CSV order: context_len, query_len, num_threads.
void sort_decoder_microbench_rows(std::vector<DecoderMicrobenchRow>* rows);

const char* decoder_microbench_cache_mode_name(DecoderMicrobenchCacheMode mode);

// Prepare synthetic tensors matching production AR/DFlash decoder step protocol.
// cache_mode=Reserved allocates past KV with spare capacity for steady append.
bool prepare_decoder_microbench_inputs(int context_len,
                                       int query_len,
                                       DecoderMicrobenchCacheMode cache_mode,
                                       ncnn::Mat* embeds,
                                       ncnn::Mat* mask,
                                       ncnn::Mat* cos,
                                       ncnn::Mat* sin,
                                       DecoderKVCache* caches,
                                       std::string* error);

// Validate production decoder outputs for one measured/warmup step.
bool validate_decoder_microbench_outputs(int context_len,
                                         int query_len,
                                         const ncnn::Mat& hidden,
                                         const DecoderKVCache& updated,
                                         const std::array<ncnn::Mat, 4>* target_hidden,
                                         std::string* error);

// Timed production run_decoder_step over fixed inputs/caches.
// Measured samples include only run_decoder_step; validate/destruct are outside.
// When HUNYUAN_OCR_ENABLE_SDPA_PROFILE is compiled in:
// - SDPA counters are reset after warmup
// - optional sdpa_profile_csv receives one CSV data row after measured repeats
bool run_decoder_microbench_cell(const ncnn::Net& decoder_net,
                                 const DecoderMicrobenchCell& cell,
                                 DecoderMicrobenchRow* row,
                                 std::string* error,
                                 std::string* sdpa_profile_csv = nullptr);

} // namespace detail
} // namespace hunyuan_ocr
