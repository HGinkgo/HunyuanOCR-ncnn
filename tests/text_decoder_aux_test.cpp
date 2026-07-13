#include "hunyuan_ocr/text_runtime.h"

#include <array>
#include <string>

int main()
{
    const char param[] =
        "7767517\n"
        "2 6\n"
        "Input in0 0 1 in0\n"
        "Split split 1 5 in0 out0 out1 out2 out3 out4\n";
    ncnn::Net net;
    if (net.load_param_mem(param) != 0)
    {
        return 1;
    }
    ncnn::Mat input(4);
    for (int i = 0; i < 4; ++i)
    {
        input[i] = static_cast<float>(i + 1);
    }
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", input) != 0)
    {
        return 2;
    }

    std::array<ncnn::Mat, 4> hidden;
    std::string error;
    if (!hunyuan_ocr::detail::extract_dflash_target_hidden(ex, &hidden, &error))
    {
        return 3;
    }
    for (const ncnn::Mat& value : hidden)
    {
        if (value.w != 4 || value[0] != 1.0f || value[3] != 4.0f)
        {
            return 4;
        }
    }

    hunyuan_ocr::TextRuntime runtime(1);
    if (runtime.dflash_ready())
    {
        return 5;
    }
    hunyuan_ocr::DFlashBlockProbeResult probe;
    error.clear();
    if (runtime.run_vlm_fixture_dflash_probe("missing", &probe, &error) ||
        error != "text runtime is not loaded")
    {
        return 6;
    }
    error.clear();
    if (runtime.load_dflash("missing", &error) || runtime.dflash_ready() || error.empty())
    {
        return 7;
    }
    return 0;
}
