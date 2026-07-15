#include "hunyuan_ocr/dflash_runtime.h"
#include "kv_cache_capacity.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

float expected_value(float salt, int channel, int row, int column)
{
    return salt + static_cast<float>(channel * 100000 + row * 100 + column);
}

void fill_rows(ncnn::Mat* mat, int absolute_row, float salt)
{
    for (int channel = 0; channel < mat->c; ++channel)
    {
        for (int row = 0; row < mat->h; ++row)
        {
            float* values = mat->channel(channel).row(row);
            for (int column = 0; column < mat->w; ++column)
            {
                values[column] = expected_value(
                    salt, channel, absolute_row + row, column);
            }
        }
    }
}

bool rows_match(const ncnn::Mat& mat,
                int mat_row,
                int rows,
                int absolute_row,
                float salt)
{
    if (mat_row < 0 || rows < 0 || mat_row + rows > mat.h)
    {
        return false;
    }
    for (int channel = 0; channel < mat.c; ++channel)
    {
        for (int row = 0; row < rows; ++row)
        {
            const float* values = mat.channel(channel).row(mat_row + row);
            for (int column = 0; column < mat.w; ++column)
            {
                if (values[column] != expected_value(
                        salt, channel, absolute_row + row, column))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

int fail(const std::string& message)
{
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main()
{
    using hunyuan_ocr::detail::allocate_kv_cache_with_capacity;
    using hunyuan_ocr::detail::append_or_grow_kv_cache;
    using hunyuan_ocr::detail::kKvCacheCapacityChunkRows;
    using hunyuan_ocr::detail::kv_cache_capacity_rows;
    using hunyuan_ocr::detail::make_kv_cache_logical_view;

    constexpr int kWidth = 8;
    constexpr int kChannels = 2;
    constexpr int kInitialRows = 120;
    constexpr int kInitialCapacity = kKvCacheCapacityChunkRows;
    constexpr int kAppendLoops = 100;
    constexpr float kKeySalt = 1.0f;
    constexpr float kValueSalt = 2.0f;
    constexpr int append_sizes[] = {1, 7, 16, 31};

    std::string error;
    ncnn::Mat key;
    ncnn::Mat value;
    if (!allocate_kv_cache_with_capacity(
            kWidth, kInitialRows, kInitialCapacity, kChannels, &key, &error) ||
        !allocate_kv_cache_with_capacity(
            kWidth, kInitialRows, kInitialCapacity, kChannels, &value, &error))
    {
        return fail(error);
    }
    fill_rows(&key, 0, kKeySalt);
    fill_rows(&value, 0, kValueSalt);

    std::vector<ncnn::Mat> stale_keys;
    std::vector<ncnn::Mat> stale_values;
    stale_keys.reserve(kAppendLoops);
    stale_values.reserve(kAppendLoops);
    int logical_rows = kInitialRows;
    int growth_count = 0;

    for (int loop = 0; loop < kAppendLoops; ++loop)
    {
        ncnn::Mat stale_key;
        ncnn::Mat stale_value;
        if (!make_kv_cache_logical_view(key, logical_rows, &stale_key, &error) ||
            !make_kv_cache_logical_view(value, logical_rows, &stale_value, &error))
        {
            return fail(error);
        }
        stale_keys.push_back(std::move(stale_key));
        stale_values.push_back(std::move(stale_value));

        const int append_rows = append_sizes[loop % 4];
        ncnn::Mat current_key(kWidth, append_rows, kChannels);
        ncnn::Mat current_value(kWidth, append_rows, kChannels);
        fill_rows(&current_key, logical_rows, kKeySalt);
        fill_rows(&current_value, logical_rows, kValueSalt);

        const void* previous_data = key.data;
        const int previous_capacity = kv_cache_capacity_rows(key);
        ncnn::Mat next_key;
        ncnn::Mat next_value;
        if (!append_or_grow_kv_cache(key,
                                     value,
                                     current_key,
                                     current_value,
                                     &next_key,
                                     &next_value,
                                     &error))
        {
            return fail(error);
        }
        logical_rows += append_rows;
        const int next_capacity = kv_cache_capacity_rows(next_key);
        if (next_capacity < logical_rows ||
            next_capacity % kKvCacheCapacityChunkRows != 0)
        {
            return fail("KV cache growth lost rounded capacity");
        }
        if (next_key.data != previous_data)
        {
            if (next_capacity <= previous_capacity)
            {
                return fail("KV cache reallocated without growing capacity");
            }
            ++growth_count;
        }
        if (!rows_match(next_key, 0, logical_rows, 0, kKeySalt) ||
            !rows_match(next_value, 0, logical_rows, 0, kValueSalt))
        {
            return fail("KV cache content changed during long append sequence");
        }
        if (!rows_match(stale_keys.back(),
                        0,
                        stale_keys.back().h,
                        0,
                        kKeySalt) ||
            !rows_match(stale_values.back(),
                        0,
                        stale_values.back().h,
                        0,
                        kValueSalt))
        {
            return fail("live shallow view changed after append or grow");
        }
        key = std::move(next_key);
        value = std::move(next_value);
    }

    if (growth_count < 10)
    {
        return fail("long append sequence did not exercise enough capacity growth");
    }

    key.release();
    value.release();
    for (size_t index = 0; index < stale_keys.size(); ++index)
    {
        if (!rows_match(stale_keys[index], 0, stale_keys[index].h, 0, kKeySalt) ||
            !rows_match(stale_values[index], 0, stale_values[index].h, 0, kValueSalt))
        {
            return fail("stale shallow view lost backing storage after cache release");
        }
    }

    constexpr int kPastRows = 8;
    constexpr int kDraftRows = 16;
    constexpr int kAcceptedRows = 5;
    constexpr int kNextRows = 4;
    ncnn::Mat past_key;
    ncnn::Mat past_value;
    if (!allocate_kv_cache_with_capacity(
            kWidth, kPastRows, 64, kChannels, &past_key, &error) ||
        !allocate_kv_cache_with_capacity(
            kWidth, kPastRows, 64, kChannels, &past_value, &error))
    {
        return fail(error);
    }
    fill_rows(&past_key, 0, 10.0f);
    fill_rows(&past_value, 0, 11.0f);
    ncnn::Mat draft_key(kWidth, kDraftRows, kChannels);
    ncnn::Mat draft_value(kWidth, kDraftRows, kChannels);
    fill_rows(&draft_key, kPastRows, 20.0f);
    fill_rows(&draft_value, kPastRows, 21.0f);

    ncnn::Mat speculative_key;
    ncnn::Mat speculative_value;
    if (!append_or_grow_kv_cache(past_key,
                                 past_value,
                                 draft_key,
                                 draft_value,
                                 &speculative_key,
                                 &speculative_value,
                                 &error))
    {
        return fail(error);
    }
    const int committed_rows = kPastRows + kAcceptedRows;
    ncnn::Mat committed_key;
    ncnn::Mat committed_value;
    if (!hunyuan_ocr::detail::view_dflash_rows(
            speculative_key, committed_rows, &committed_key, &error) ||
        !hunyuan_ocr::detail::view_dflash_rows(
            speculative_value, committed_rows, &committed_value, &error))
    {
        return fail(error);
    }

    past_key.release();
    past_value.release();
    speculative_key.release();
    speculative_value.release();
    if (!rows_match(committed_key, 0, kPastRows, 0, 10.0f) ||
        !rows_match(committed_key, kPastRows, kAcceptedRows, kPastRows, 20.0f) ||
        !rows_match(committed_value, 0, kPastRows, 0, 11.0f) ||
        !rows_match(committed_value, kPastRows, kAcceptedRows, kPastRows, 21.0f))
    {
        return fail("DFlash committed view did not retain speculative backing");
    }

    ncnn::Mat next_key(kWidth, kNextRows, kChannels);
    ncnn::Mat next_value(kWidth, kNextRows, kChannels);
    fill_rows(&next_key, committed_rows, 30.0f);
    fill_rows(&next_value, committed_rows, 31.0f);
    ncnn::Mat appended_key;
    ncnn::Mat appended_value;
    if (!append_or_grow_kv_cache(committed_key,
                                 committed_value,
                                 next_key,
                                 next_value,
                                 &appended_key,
                                 &appended_value,
                                 &error))
    {
        return fail(error);
    }
    if (appended_key.data != committed_key.data ||
        appended_value.data != committed_value.data ||
        !rows_match(appended_key, committed_rows, kNextRows, committed_rows, 30.0f) ||
        !rows_match(appended_value, committed_rows, kNextRows, committed_rows, 31.0f))
    {
        return fail("DFlash re-append did not overwrite the rejected draft zone");
    }

    appended_key.release();
    appended_value.release();
    if (!rows_match(committed_key, 0, kPastRows, 0, 10.0f) ||
        !rows_match(committed_key, kPastRows, kAcceptedRows, kPastRows, 20.0f) ||
        !rows_match(committed_value, 0, kPastRows, 0, 11.0f) ||
        !rows_match(committed_value, kPastRows, kAcceptedRows, kPastRows, 21.0f))
    {
        return fail("DFlash committed view became invalid after appended cache release");
    }

    std::cout << "append_loops=" << kAppendLoops << '\n'
              << "final_logical_rows=" << logical_rows << '\n'
              << "capacity_growth_count=" << growth_count << '\n'
              << "retained_shallow_views=" << stale_keys.size() << '\n';
    return 0;
}
