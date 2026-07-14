#include "decoder_microbench.h"
#include "precise_sdpa_profile.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const char* message)
{
    std::cerr << message << '\n';
    return 1;
}

bool near(double value, double expected, double tol = 1e-9)
{
    return std::fabs(value - expected) <= tol;
}

} // namespace

int main()
{
    using hunyuan_ocr::detail::DecoderMicrobenchCell;
    using hunyuan_ocr::detail::DecoderMicrobenchRow;
    using hunyuan_ocr::detail::DecoderMicrobenchTiming;
    using hunyuan_ocr::detail::compute_timing_stats;
    using hunyuan_ocr::detail::decoder_microbench_csv_header;
    using hunyuan_ocr::detail::default_decoder_microbench_matrix;
    using hunyuan_ocr::detail::expand_decoder_microbench_matrix;
    using hunyuan_ocr::detail::format_decoder_microbench_csv_row;
    using hunyuan_ocr::detail::parse_positive_int_list;

    std::string error;
    std::vector<int> values;
    if (!parse_positive_int_list("4,8,8,16", &values, &error) ||
        values != std::vector<int>({4, 8, 16}))
    {
        return fail("parse_positive_int_list should dedupe while preserving order");
    }
    error.clear();
    if (parse_positive_int_list("4,-1,8", &values, &error) || error.empty())
    {
        return fail("negative values must be rejected");
    }
    error.clear();
    if (parse_positive_int_list("0,4", &values, &error) || error.empty())
    {
        return fail("zero values must be rejected");
    }
    error.clear();
    if (parse_positive_int_list("4x", &values, &error) || error.empty())
    {
        return fail("partial integer token 4x must be rejected");
    }
    error.clear();
    if (parse_positive_int_list("8foo", &values, &error) || error.empty())
    {
        return fail("partial integer token 8foo must be rejected");
    }
    error.clear();
    if (parse_positive_int_list("4,8foo,16", &values, &error) || error.empty())
    {
        return fail("list containing partial integer token must be rejected");
    }
    error.clear();
    // 4294967297 wraps to 1 under unchecked 32-bit multiply-add and must not
    // be accepted as a positive integer token.
    if (parse_positive_int_list("4294967297", &values, &error) || error.empty())
    {
        return fail("overflowing integer token must be rejected");
    }
    error.clear();
    if (parse_positive_int_list("4,4294967297,16", &values, &error) || error.empty())
    {
        return fail("list containing overflowing integer token must be rejected");
    }
    error.clear();
    if (parse_positive_int_list("999999999999999999999", &values, &error) ||
        error.empty())
    {
        return fail("very large overflowing integer token must be rejected");
    }

    const std::vector<double> samples = {5.0, 1.0, 3.0, 9.0, 7.0};
    const DecoderMicrobenchTiming stats = compute_timing_stats(samples, 16);
    if (!near(stats.min_ms, 1.0) ||
        !near(stats.mean_ms, 5.0) ||
        !near(stats.median_ms, 5.0) ||
        !near(stats.per_query_token_ms, 5.0 / 16.0))
    {
        return fail("timing stats mismatch");
    }

    const std::vector<double> even = {4.0, 1.0, 3.0, 2.0};
    const DecoderMicrobenchTiming even_stats = compute_timing_stats(even, 1);
    if (!near(even_stats.median_ms, 2.5))
    {
        return fail("even-length median should average the middle pair");
    }

    error.clear();
    const auto matrix = expand_decoder_microbench_matrix(
        {256, 512}, {1, 16}, {4, 8}, 2, 5, &error);
    if (!error.empty() || matrix.size() != 8)
    {
        return fail("matrix expansion size mismatch");
    }
    if (matrix[0].context_len != 256 || matrix[0].query_len != 1 ||
        matrix[0].num_threads != 4 || matrix[0].warmup != 2 || matrix[0].repeat != 5)
    {
        return fail("matrix expansion order mismatch at first cell");
    }
    if (matrix[1].context_len != 256 || matrix[1].query_len != 1 ||
        matrix[1].num_threads != 8)
    {
        return fail("matrix expansion order mismatch at second cell");
    }
    if (matrix[2].context_len != 256 || matrix[2].query_len != 16 ||
        matrix[2].num_threads != 4)
    {
        return fail("matrix expansion order mismatch at third cell");
    }
    if (matrix.back().context_len != 512 || matrix.back().query_len != 16 ||
        matrix.back().num_threads != 8)
    {
        return fail("matrix expansion order mismatch at last cell");
    }

    const auto defaults = default_decoder_microbench_matrix(2, 5);
    if (defaults.size() != 24)
    {
        return fail("default matrix must contain 24 cells");
    }

    DecoderMicrobenchRow row;
    row.cell.context_len = 256;
    row.cell.query_len = 16;
    row.cell.num_threads = 32;
    row.cell.warmup = 2;
    row.cell.repeat = 5;
    row.timing.min_ms = 1.25;
    row.timing.mean_ms = 1.5;
    row.timing.median_ms = 1.4;
    row.timing.per_query_token_ms = 1.4 / 16.0;
    const std::string header = decoder_microbench_csv_header();
    const std::string csv = format_decoder_microbench_csv_row(row);
    if (header !=
        "context_len,query_len,num_threads,warmup,repeat,cache_mode,min_ms,mean_ms,median_ms,per_query_token_ms")
    {
        return fail("unexpected CSV header schema");
    }
    if (csv.find("256,16,32,2,5,exact,") != 0)
    {
        return fail("CSV row prefix mismatch");
    }
    if (csv.find("1.400") == std::string::npos)
    {
        return fail("CSV row should include median with fixed precision");
    }

    // PreciseSDPA profile CSV schema (alloc/copy split under kv_concat).
    {
        using hunyuan_ocr::detail::PreciseSdpaProfileSnapshot;
        using hunyuan_ocr::detail::format_precise_sdpa_profile_csv_row;
        using hunyuan_ocr::detail::precise_sdpa_profile_csv_header;

        const std::string sdpa_header = precise_sdpa_profile_csv_header();
        if (sdpa_header.find("kv_alloc_ms") == std::string::npos ||
            sdpa_header.find("kv_copy_ms") == std::string::npos ||
            sdpa_header.find("kv_concat_ms") == std::string::npos)
        {
            return fail("SDPA profile CSV header must include kv_alloc_ms, kv_copy_ms, and kv_concat_ms");
        }
        // Stable column order: ... total_ms,kv_concat_ms,kv_alloc_ms,kv_copy_ms,qk_score_ms ...
        if (sdpa_header.find(
                "call_count,total_ms,kv_concat_ms,kv_alloc_ms,kv_copy_ms,qk_score_ms") ==
            std::string::npos)
        {
            return fail("SDPA profile CSV header column order mismatch for kv split fields");
        }

        PreciseSdpaProfileSnapshot snap;
        snap.call_count = 168;
        snap.total_ms = 100.0;
        snap.kv_alloc_ms = 12.5;
        snap.kv_copy_ms = 37.5;
        snap.kv_concat_ms = 50.0; // must remain alloc+copy for compatibility
        snap.qk_score_ms = 20.0;
        snap.softmax_ms = 1.0;
        snap.pv_ms = 25.0;
        snap.output_alloc_ms = 0.5;
        snap.other_ms = 3.5;
        snap.past_kv_copy_bytes = 1024;
        snap.current_kv_copy_bytes = 256;
        snap.last_query_len = 16;
        snap.last_key_len = 1040;
        snap.last_num_heads = 16;
        snap.last_num_threads = 16;
        const std::string sdpa_row = format_precise_sdpa_profile_csv_row(
            1024, 16, 16, 3, 7, 12.345, snap);
        if (sdpa_row.find("1024,16,16,3,7,") != 0)
        {
            return fail("SDPA profile CSV row prefix mismatch");
        }
        if (sdpa_row.find(",50.000,12.500,37.500,") == std::string::npos)
        {
            return fail("SDPA profile CSV row must emit kv_concat,kv_alloc,kv_copy in order");
        }
    }

    return 0;
}
