#include "decoder_microbench.h"

#include "hunyuan_ocr/dflash_runtime.h"
#include "kv_cache_capacity.h"
#include "precise_sdpa_profile.h"
#include "text_decoder_step.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <memory>
#include <sstream>

namespace hunyuan_ocr {
namespace detail {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool has_non_finite(const ncnn::Mat& mat)
{
    if (mat.empty() || mat.elemsize != 4u)
    {
        return true;
    }
    const int h = mat.h > 0 ? mat.h : 1;
    const int c = mat.c > 0 ? mat.c : 1;
    for (int channel = 0; channel < c; ++channel)
    {
        for (int row = 0; row < h; ++row)
        {
            const float* values = mat.channel(channel).row(row);
            for (int col = 0; col < mat.w; ++col)
            {
                if (!std::isfinite(values[col]))
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool fill_deterministic(ncnn::Mat* mat, float seed)
{
    if (mat == nullptr || mat->empty() || mat->elemsize != 4u || mat->elempack != 1)
    {
        return false;
    }
    const int h = mat->h > 0 ? mat->h : 1;
    const int c = mat->c > 0 ? mat->c : 1;
    for (int channel = 0; channel < c; ++channel)
    {
        for (int row = 0; row < h; ++row)
        {
            float* values = mat->channel(channel).row(row);
            for (int col = 0; col < mat->w; ++col)
            {
                const float phase =
                    seed + 0.017f * static_cast<float>(channel) +
                    0.031f * static_cast<float>(row) + 0.007f * static_cast<float>(col);
                values[col] = 0.01f * std::sin(phase);
            }
        }
    }
    return true;
}

bool validate_cache_shape(const ncnn::Mat& cache,
                          int expected_rows,
                          const char* label,
                          std::string* error)
{
    if (cache.dims != 3 ||
        cache.w != kTextHeadDim ||
        cache.h != expected_rows ||
        cache.c != kTextCacheHeads ||
        cache.elemsize != 4u ||
        cache.elempack != 1)
    {
        if (error)
        {
            *error = std::string("invalid ") + label + " shape";
        }
        return false;
    }
    if (has_non_finite(cache))
    {
        if (error) *error = std::string("non-finite values in ") + label;
        return false;
    }
    return true;
}

bool parse_strict_positive_int(const std::string& item, int* value, std::string* error)
{
    if (value == nullptr)
    {
        if (error) *error = "value pointer is null";
        return false;
    }
    if (item.empty())
    {
        if (error) *error = "empty integer token";
        return false;
    }
    size_t index = 0;
    if (item[0] == '+')
    {
        index = 1;
    }
    if (index >= item.size())
    {
        if (error) *error = "failed to parse integer list: " + item;
        return false;
    }
    int parsed = 0;
    for (; index < item.size(); ++index)
    {
        const unsigned char ch = static_cast<unsigned char>(item[index]);
        if (!std::isdigit(ch))
        {
            if (error) *error = "failed to parse integer list: " + item;
            return false;
        }
        const int digit = item[index] - '0';
        if (parsed > (INT_MAX - digit) / 10)
        {
            if (error) *error = "integer token overflows int range: " + item;
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    if (parsed <= 0)
    {
        if (error) *error = "values must be positive integers";
        return false;
    }
    *value = parsed;
    return true;
}

} // namespace

std::vector<DecoderMicrobenchCell> default_decoder_microbench_matrix(int warmup, int repeat)
{
    std::string error;
    return expand_decoder_microbench_matrix(
        {256, 512, 1024}, {1, 16}, {4, 8, 16, 32}, warmup, repeat, &error);
}

bool parse_positive_int_list(const std::string& text,
                             std::vector<int>* values,
                             std::string* error)
{
    if (values == nullptr)
    {
        if (error) *error = "values pointer is null";
        return false;
    }
    values->clear();
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (item.empty())
        {
            continue;
        }
        int value = 0;
        if (!parse_strict_positive_int(item, &value, error))
        {
            return false;
        }
        if (std::find(values->begin(), values->end(), value) == values->end())
        {
            values->push_back(value);
        }
    }
    if (values->empty())
    {
        if (error) *error = "integer list is empty";
        return false;
    }
    return true;
}

DecoderMicrobenchTiming compute_timing_stats(const std::vector<double>& samples_ms,
                                             int query_len)
{
    DecoderMicrobenchTiming timing;
    if (samples_ms.empty() || query_len <= 0)
    {
        return timing;
    }
    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    timing.min_ms = sorted.front();
    double sum = 0.0;
    for (double value : sorted)
    {
        sum += value;
    }
    timing.mean_ms = sum / static_cast<double>(sorted.size());
    if (sorted.size() % 2 == 1)
    {
        timing.median_ms = sorted[sorted.size() / 2];
    }
    else
    {
        timing.median_ms =
            0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    }
    timing.per_query_token_ms = timing.median_ms / static_cast<double>(query_len);
    return timing;
}

const char* decoder_microbench_cache_mode_name(DecoderMicrobenchCacheMode mode)
{
    return mode == DecoderMicrobenchCacheMode::Reserved ? "reserved" : "exact";
}

std::string decoder_microbench_csv_header()
{
    return "context_len,query_len,num_threads,warmup,repeat,cache_mode,min_ms,mean_ms,median_ms,per_query_token_ms";
}

std::string format_decoder_microbench_csv_row(const DecoderMicrobenchRow& row)
{
    std::ostringstream stream;
    stream << row.cell.context_len << ','
           << row.cell.query_len << ','
           << row.cell.num_threads << ','
           << row.cell.warmup << ','
           << row.cell.repeat << ','
           << decoder_microbench_cache_mode_name(row.cell.cache_mode) << ','
           << std::fixed << std::setprecision(3)
           << row.timing.min_ms << ','
           << row.timing.mean_ms << ','
           << row.timing.median_ms << ','
           << row.timing.per_query_token_ms;
    return stream.str();
}

std::vector<DecoderMicrobenchCell> expand_decoder_microbench_matrix(
    const std::vector<int>& context_lens,
    const std::vector<int>& query_lens,
    const std::vector<int>& thread_counts,
    int warmup,
    int repeat,
    std::string* error)
{
    std::vector<DecoderMicrobenchCell> matrix;
    if (context_lens.empty() || query_lens.empty() || thread_counts.empty())
    {
        if (error) *error = "matrix dimensions must be non-empty";
        return matrix;
    }
    if (warmup < 0 || repeat <= 0)
    {
        if (error) *error = "warmup must be non-negative and repeat must be positive";
        return matrix;
    }
    matrix.reserve(context_lens.size() * query_lens.size() * thread_counts.size());
    for (int context_len : context_lens)
    {
        for (int query_len : query_lens)
        {
            for (int num_threads : thread_counts)
            {
                if (context_len <= 0 || query_len <= 0 || num_threads <= 0)
                {
                    if (error) *error = "matrix values must be positive";
                    matrix.clear();
                    return matrix;
                }
                DecoderMicrobenchCell cell;
                cell.context_len = context_len;
                cell.query_len = query_len;
                cell.num_threads = num_threads;
                cell.warmup = warmup;
                cell.repeat = repeat;
                matrix.push_back(cell);
            }
        }
    }
    return matrix;
}

void sort_decoder_microbench_rows(std::vector<DecoderMicrobenchRow>* rows)
{
    if (rows == nullptr)
    {
        return;
    }
    std::sort(rows->begin(), rows->end(),
              [](const DecoderMicrobenchRow& left, const DecoderMicrobenchRow& right) {
                  if (left.cell.context_len != right.cell.context_len)
                  {
                      return left.cell.context_len < right.cell.context_len;
                  }
                  if (left.cell.query_len != right.cell.query_len)
                  {
                      return left.cell.query_len < right.cell.query_len;
                  }
                  return left.cell.num_threads < right.cell.num_threads;
              });
}

bool prepare_decoder_microbench_inputs(int context_len,
                                       int query_len,
                                       DecoderMicrobenchCacheMode cache_mode,
                                       ncnn::Mat* embeds,
                                       ncnn::Mat* mask,
                                       ncnn::Mat* cos,
                                       ncnn::Mat* sin,
                                       DecoderKVCache* caches,
                                       std::string* error)
{
    if (embeds == nullptr || mask == nullptr || cos == nullptr || sin == nullptr ||
        caches == nullptr)
    {
        if (error) *error = "microbench input pointers are null";
        return false;
    }
    if (context_len <= 0 || query_len <= 0)
    {
        if (error) *error = "context_len and query_len must be positive";
        return false;
    }

    embeds->create(kTextHiddenSize, query_len);
    mask->create(context_len + query_len, query_len);
    cos->create(kTextHeadDim, query_len);
    sin->create(kTextHeadDim, query_len);
    if (embeds->empty() || mask->empty() || cos->empty() || sin->empty())
    {
        if (error) *error = "failed to allocate microbench tensors";
        return false;
    }
    if (!fill_deterministic(embeds, 1.0f) ||
        !fill_deterministic(cos, 2.0f) ||
        !fill_deterministic(sin, 3.0f))
    {
        if (error) *error = "failed to fill microbench tensors";
        return false;
    }
    mask->fill(0.0f);
    if (query_len > 1)
    {
        for (int row = 0; row < query_len; ++row)
        {
            float* values = mask->row(row);
            for (int col = context_len + row + 1; col < context_len + query_len; ++col)
            {
                values[col] = -1.0e38f;
            }
        }
    }

    caches->clear();
    caches->reserve(kTextAttentionLayerCount);
    // Reserved mode leaves room for at least one DFlash block append without grow.
    const int reserved_capacity =
        round_up_kv_capacity_rows(context_len + std::max(query_len, kDFlashBlockSize));
    for (int layer = 0; layer < kTextAttentionLayerCount; ++layer)
    {
        ncnn::Mat key;
        ncnn::Mat value;
        if (cache_mode == DecoderMicrobenchCacheMode::Reserved)
        {
            if (!allocate_kv_cache_with_capacity(kTextHeadDim,
                                                 context_len,
                                                 reserved_capacity,
                                                 kTextCacheHeads,
                                                 &key,
                                                 error) ||
                !allocate_kv_cache_with_capacity(kTextHeadDim,
                                                 context_len,
                                                 reserved_capacity,
                                                 kTextCacheHeads,
                                                 &value,
                                                 error))
            {
                return false;
            }
        }
        else
        {
            key.create(kTextHeadDim, context_len, kTextCacheHeads);
            value.create(kTextHeadDim, context_len, kTextCacheHeads);
        }
        if (key.empty() || value.empty() ||
            !fill_deterministic(&key, 10.0f + static_cast<float>(layer)) ||
            !fill_deterministic(&value, 20.0f + static_cast<float>(layer)))
        {
            if (error) *error = "failed to allocate microbench KV cache";
            return false;
        }
        caches->emplace_back(std::move(key), std::move(value));
    }
    return true;
}

bool validate_decoder_microbench_outputs(int context_len,
                                         int query_len,
                                         const ncnn::Mat& hidden,
                                         const DecoderKVCache& updated,
                                         const std::array<ncnn::Mat, 4>* target_hidden,
                                         std::string* error)
{
    if (hidden.dims != 2 ||
        hidden.w != kTextHiddenSize ||
        hidden.h != query_len ||
        hidden.elemsize != 4u ||
        hidden.elempack != 1 ||
        has_non_finite(hidden))
    {
        if (error) *error = "invalid decoder hidden output";
        return false;
    }
    if (static_cast<int>(updated.size()) != kTextAttentionLayerCount)
    {
        if (error) *error = "updated KV cache layer count mismatch";
        return false;
    }
    const int expected_rows = context_len + query_len;
    for (int layer = 0; layer < kTextAttentionLayerCount; ++layer)
    {
        if (!validate_cache_shape(updated[static_cast<size_t>(layer)].first,
                                  expected_rows,
                                  "out_cache_k",
                                  error) ||
            !validate_cache_shape(updated[static_cast<size_t>(layer)].second,
                                  expected_rows,
                                  "out_cache_v",
                                  error))
        {
            return false;
        }
    }
    if (query_len == kDFlashBlockSize)
    {
        if (target_hidden == nullptr)
        {
            if (error) *error = "target hidden required for query_len=16";
            return false;
        }
        for (size_t index = 0; index < target_hidden->size(); ++index)
        {
            const ncnn::Mat& layer_hidden = (*target_hidden)[index];
            if (layer_hidden.dims != 2 ||
                layer_hidden.w != kTextHiddenSize ||
                layer_hidden.h != query_len ||
                layer_hidden.elemsize != 4u ||
                layer_hidden.elempack != 1 ||
                has_non_finite(layer_hidden))
            {
                if (error)
                {
                    *error = "invalid target hidden out" + std::to_string(index + 1);
                }
                return false;
            }
        }
    }
    return true;
}

bool run_decoder_microbench_cell(const ncnn::Net& decoder_net,
                                 const DecoderMicrobenchCell& cell,
                                 DecoderMicrobenchRow* row,
                                 std::string* error,
                                 std::string* sdpa_profile_csv)
{
    if (row == nullptr)
    {
        if (error) *error = "result row pointer is null";
        return false;
    }
    if (cell.context_len <= 0 || cell.query_len <= 0 || cell.num_threads <= 0 ||
        cell.warmup < 0 || cell.repeat <= 0)
    {
        if (error) *error = "invalid microbench cell";
        return false;
    }

    ncnn::Mat embeds;
    ncnn::Mat mask;
    ncnn::Mat cos;
    ncnn::Mat sin;
    DecoderKVCache caches;
    if (!prepare_decoder_microbench_inputs(cell.context_len,
                                           cell.query_len,
                                           cell.cache_mode,
                                           &embeds,
                                           &mask,
                                           &cos,
                                           &sin,
                                           &caches,
                                           error))
    {
        return false;
    }

    const bool need_target_hidden = cell.query_len == kDFlashBlockSize;

    for (int index = 0; index < cell.warmup; ++index)
    {
        ncnn::Mat hidden;
        DecoderKVCache updated;
        std::array<ncnn::Mat, 4> target_hidden;
        std::array<ncnn::Mat, 4>* target_ptr =
            need_target_hidden ? &target_hidden : nullptr;
        if (!run_decoder_step(decoder_net,
                              embeds,
                              mask,
                              cos,
                              sin,
                              caches,
                              &hidden,
                              &updated,
                              target_ptr,
                              error))
        {
            return false;
        }
        if (!validate_decoder_microbench_outputs(cell.context_len,
                                                 cell.query_len,
                                                 hidden,
                                                 updated,
                                                 target_ptr,
                                                 error))
        {
            return false;
        }
    }

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
    reset_precise_sdpa_profile();
#endif

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(cell.repeat));
    for (int index = 0; index < cell.repeat; ++index)
    {
        // Output containers live outside the timed region so their destructors
        // cannot pollute measured samples.
        auto hidden = std::make_unique<ncnn::Mat>();
        auto updated = std::make_unique<DecoderKVCache>();
        std::unique_ptr<std::array<ncnn::Mat, 4>> target_hidden;
        std::array<ncnn::Mat, 4>* target_ptr = nullptr;
        if (need_target_hidden)
        {
            target_hidden = std::make_unique<std::array<ncnn::Mat, 4>>();
            target_ptr = target_hidden.get();
        }

        const auto start = Clock::now();
        const bool ok = run_decoder_step(decoder_net,
                                         embeds,
                                         mask,
                                         cos,
                                         sin,
                                         caches,
                                         hidden.get(),
                                         updated.get(),
                                         target_ptr,
                                         error);
        const auto end = Clock::now();
        if (!ok)
        {
            return false;
        }
        samples.push_back(elapsed_ms(start, end));

        if (!validate_decoder_microbench_outputs(cell.context_len,
                                                 cell.query_len,
                                                 *hidden,
                                                 *updated,
                                                 target_ptr,
                                                 error))
        {
            return false;
        }
        // Explicitly release after validation, still outside the timed region.
        target_hidden.reset();
        updated.reset();
        hidden.reset();
    }

    row->cell = cell;
    row->timing = compute_timing_stats(samples, cell.query_len);

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
    const PreciseSdpaProfileSnapshot snapshot = snapshot_precise_sdpa_profile();
    const std::uint64_t expected_calls =
        static_cast<std::uint64_t>(kTextAttentionLayerCount) *
        static_cast<std::uint64_t>(cell.repeat);
    if (snapshot.call_count != expected_calls)
    {
        if (error)
        {
            *error = "SDPA call_count mismatch: got " +
                     std::to_string(snapshot.call_count) + ", expected " +
                     std::to_string(expected_calls) +
                     " (24 layers * repeat)";
        }
        return false;
    }
    if (sdpa_profile_csv != nullptr)
    {
        *sdpa_profile_csv = format_precise_sdpa_profile_csv_row(
            cell.context_len,
            cell.query_len,
            cell.num_threads,
            cell.warmup,
            cell.repeat,
            row->timing.median_ms,
            snapshot);
    }
#else
    if (sdpa_profile_csv != nullptr)
    {
        sdpa_profile_csv->clear();
    }
#endif
    return true;
}

} // namespace detail
} // namespace hunyuan_ocr
