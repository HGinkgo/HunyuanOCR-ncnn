#include "hunyuan_ocr/dflash_runtime.h"

#include <cmath>
#include <string>
#include <vector>

int main()
{
    const std::vector<int> block = hunyuan_ocr::detail::build_dflash_block(42);
    if (block.size() != 16 || block[0] != 42)
    {
        return 1;
    }
    for (size_t i = 1; i < block.size(); ++i)
    {
        if (block[i] != 120817)
        {
            return 2;
        }
    }

    const int past_len = 3;
    const std::vector<float> verify_mask =
        hunyuan_ocr::detail::build_dflash_verify_mask(past_len);
    const int key_len = past_len + 16;
    if (verify_mask.size() != static_cast<size_t>(16 * key_len))
    {
        return 3;
    }
    for (int row = 0; row < 16; ++row)
    {
        for (int col = 0; col < key_len; ++col)
        {
            const float value = verify_mask[static_cast<size_t>(row) * key_len + col];
            const bool visible = col <= past_len + row;
            if ((visible && value != 0.0f) || (!visible && value > -1.0e30f))
            {
                return 4;
            }
        }
    }

    const std::vector<int> proposed = {10, 11, 12, 13};
    const std::vector<int> posterior = {11, 12, 99, 100};
    if (hunyuan_ocr::detail::dflash_acceptance_length(proposed, posterior) != 2)
    {
        return 5;
    }
    if (hunyuan_ocr::detail::dflash_acceptance_length({10, 11}, {11}) != -1)
    {
        return 6;
    }

    const std::vector<float> draft_cos = hunyuan_ocr::detail::build_dflash_rope(2, true);
    if (draft_cos.size() != 256)
    {
        return 7;
    }
    const float expected_cos_dim2 = std::cos(std::pow(10000.0f, -4.0f / 128.0f));
    if (std::fabs(draft_cos[128 + 2] - expected_cos_dim2) > 1.0e-7f)
    {
        return 8;
    }
    const std::vector<float> draft_mask = hunyuan_ocr::detail::build_dflash_attention_mask(3);
    if (draft_mask.size() != static_cast<size_t>(16 * 19))
    {
        return 9;
    }
    for (float value : draft_mask)
    {
        if (value != 0.0f)
        {
            return 10;
        }
    }

    hunyuan_ocr::DFlashDraftRuntime runtime(1);
    if (runtime.ready())
    {
        return 11;
    }
    hunyuan_ocr::DFlashDraftInput input;
    ncnn::Mat output;
    std::string error;
    if (runtime.run(input, &output, &error) || error != "DFlash draft runtime is not loaded")
    {
        return 12;
    }
    error.clear();
    if (runtime.load("missing.param", "missing.bin", &error) || runtime.ready() || error.empty())
    {
        return 13;
    }

    ncnn::Mat cache(2, 4, 2);
    for (int channel = 0; channel < 2; ++channel)
    {
        for (int row = 0; row < 4; ++row)
        {
            float* values = cache.channel(channel).row(row);
            values[0] = static_cast<float>(channel * 100 + row * 10 + 1);
            values[1] = static_cast<float>(channel * 100 + row * 10 + 2);
        }
    }
    ncnn::Mat view;
    error.clear();
    if (!hunyuan_ocr::detail::view_dflash_rows(cache, 3, &view, &error) ||
        view.dims != 3 ||
        view.w != 2 ||
        view.h != 3 ||
        view.c != 2 ||
        view.data != cache.data ||
        view.cstep != cache.cstep ||
        view.nstep != cache.nstep ||
        cache.h != 4 ||
        view.channel(1).row(2)[1] != 122.0f)
    {
        return 14;
    }
    cache.channel(1).row(2)[1] = 222.0f;
    if (view.channel(1).row(2)[1] != 222.0f)
    {
        return 15;
    }
    if (hunyuan_ocr::detail::view_dflash_rows(cache, 0, &view, &error) ||
        hunyuan_ocr::detail::view_dflash_rows(cache, 5, &view, &error))
    {
        return 16;
    }
    ncnn::Mat flat(8);
    if (hunyuan_ocr::detail::view_dflash_rows(flat, 1, &view, &error))
    {
        return 17;
    }
    ncnn::Mat image(2, 4);
    if (hunyuan_ocr::detail::view_dflash_rows(image, 2, &view, &error))
    {
        return 18;
    }

    ncnn::Mat context(2, 2);
    context.row(0)[0] = 1.0f;
    context.row(0)[1] = 2.0f;
    context.row(1)[0] = 3.0f;
    context.row(1)[1] = 4.0f;
    ncnn::Mat additions(2, 3);
    additions.row(0)[0] = 5.0f;
    additions.row(0)[1] = 6.0f;
    additions.row(1)[0] = 7.0f;
    additions.row(1)[1] = 8.0f;
    additions.row(2)[0] = 9.0f;
    additions.row(2)[1] = 10.0f;
    error.clear();
    if (!hunyuan_ocr::detail::append_dflash_rows(&context, additions, 2, &error) ||
        context.w != 2 || context.h != 4 ||
        context.row(2)[0] != 5.0f || context.row(3)[1] != 8.0f)
    {
        return 19;
    }
    return 0;
}
