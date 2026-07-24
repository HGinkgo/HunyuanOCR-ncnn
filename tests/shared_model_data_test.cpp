#include "mapped_model_file.h"

#include <net.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool nearly_equal(float actual, float expected)
{
    return std::fabs(actual - expected) < 1.0e-6f;
}

const char* embed_param()
{
    return "7767517\n"
           "2 2\n"
           "Input in0 0 1 in0\n"
           "Embed embed_0 1 1 in0 out0 0=2 1=2 2=0 3=4\n";
}

const char* lm_head_param()
{
    return "7767517\n"
           "2 2\n"
           "Input in0 0 1 in0\n"
           "Gemm gemm_0 1 1 in0 out0 10=-1 2=0 3=1 4=0 5=1 6=1 7=1 8=2 9=2\n";
}

bool configure_net(ncnn::Net* net, const char* param)
{
    net->opt.lightmode = false;
    net->opt.num_threads = 1;
    net->opt.use_packing_layout = false;
    return net->load_param_mem(param) == 0;
}

} // namespace

int main()
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "hunyuan_ocr_shared_model_data_test";
    std::error_code filesystem_error;
    std::filesystem::remove_all(root, filesystem_error);
    filesystem_error.clear();
    std::filesystem::create_directories(root, filesystem_error);
    if (!expect(!filesystem_error, "failed to create shared-data test directory"))
    {
        return 1;
    }

    const std::filesystem::path weights_path = root / "tied_weights.ncnn.bin";
    const std::uint32_t fp32_tag = 0;
    const std::array<float, 4> weights{{1.0f, 2.0f, 3.0f, 4.0f}};
    {
        std::ofstream output(weights_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(&fp32_tag), sizeof(fp32_tag));
        output.write(reinterpret_cast<const char*>(weights.data()),
                     static_cast<std::streamsize>(sizeof(weights)));
    }

    std::string error;
    hunyuan_ocr::SharedModelData shared;
    if (!expect(shared.open(weights_path, false, &error),
                "owned shared model data must open") ||
        !expect(shared.ready(), "owned shared model data must be ready") ||
        !expect(shared.size() == sizeof(fp32_tag) + sizeof(weights),
                "owned shared model data size mismatch") ||
        !expect(shared.mapped_bytes() == 0,
                "owned shared model data must not report mapped bytes"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 2;
    }

    ncnn::Net embed_net;
    ncnn::Net lm_head_net;
    const unsigned char* const shared_data = shared.data();
    if (!expect(configure_net(&embed_net, embed_param()), "embed param load failed") ||
        !expect(configure_net(&lm_head_net, lm_head_param()), "lm_head param load failed") ||
        !expect(hunyuan_ocr::load_shared_model_data(embed_net, shared, &error),
                "embed shared weight load failed") ||
        !expect(hunyuan_ocr::load_shared_model_data(lm_head_net, shared, &error),
                "lm_head shared weight load failed") ||
        !expect(shared.data() == shared_data,
                "loading both nets must preserve the shared backing"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 3;
    }

    ncnn::Mat token_id(1, static_cast<size_t>(4u));
    static_cast<int*>(token_id.data)[0] = 1;
    ncnn::Mat embedding;
    ncnn::Extractor embed_extractor = embed_net.create_extractor();
    if (!expect(embed_extractor.input("in0", token_id) == 0, "embed input failed") ||
        !expect(embed_extractor.extract("out0", embedding) == 0, "embed extract failed") ||
        !expect(embedding.w == 2 && embedding.h == 1, "embed output shape mismatch") ||
        !expect(nearly_equal(embedding[0], 3.0f) && nearly_equal(embedding[1], 4.0f),
                "embed output mismatch"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 5;
    }

    ncnn::Mat hidden(2);
    hidden[0] = 1.0f;
    hidden[1] = 1.0f;
    ncnn::Mat logits;
    ncnn::Extractor lm_head_extractor = lm_head_net.create_extractor();
    if (!expect(lm_head_extractor.input("in0", hidden) == 0, "lm_head input failed") ||
        !expect(lm_head_extractor.extract("out0", logits) == 0, "lm_head extract failed") ||
        !expect(logits.w == 2 && logits.h == 1, "lm_head output shape mismatch") ||
        !expect(nearly_equal(logits[0], 3.0f) && nearly_equal(logits[1], 7.0f),
                "lm_head output mismatch"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 6;
    }

    hunyuan_ocr::SharedModelData mapped;
    error.clear();
    if (!expect(mapped.open(weights_path, true, &error),
                "mapped shared model data must open") ||
        !expect(mapped.ready(), "mapped shared model data must be ready") ||
        !expect(mapped.mapped_bytes() == mapped.size(),
                "mapped shared model bytes must be counted once"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 7;
    }

    std::filesystem::remove_all(root, filesystem_error);
    std::cout << "shared Embed/Gemm model backing passed\n";
    return 0;
}
