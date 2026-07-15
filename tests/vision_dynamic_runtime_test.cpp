#include "hunyuan_ocr/utf8.h"
#include "hunyuan_ocr/vision_runtime.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kPatchDim = 3 * 16 * 16;
constexpr int kHiddenSize = 1024;
constexpr int kMergeSize = 2;
constexpr int kInferenceLoops = 10;

struct GridShape {
    int height;
    int width;
};

bool features_are_finite(const std::vector<float>& values)
{
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: vision_dynamic_runtime_test DYNAMIC_VISION_DIR\n";
        return 1;
    }

    const std::filesystem::path model_dir = hunyuan_ocr::path_from_utf8(argv[1]);
    const std::string param_path =
        hunyuan_ocr::path_to_utf8(model_dir / "vision.ncnn.param");
    const std::string bin_path =
        hunyuan_ocr::path_to_utf8(model_dir / "vision.ncnn.bin");
    const std::string pos_embed_path =
        hunyuan_ocr::path_to_utf8(model_dir / "pos_embed.bin");

    hunyuan_ocr::VisionRuntime runtime(2);
    std::string error;
    if (!runtime.load_dynamic(param_path, bin_path, pos_embed_path, &error))
    {
        std::cerr << error << '\n';
        return 2;
    }

    const GridShape shapes[] = {{4, 6}, {6, 8}};
    for (int loop = 0; loop < kInferenceLoops; ++loop)
    {
        const GridShape shape = shapes[loop % 2];
        const int patch_count = shape.height * shape.width;
        std::vector<float> pixels(
            static_cast<size_t>(patch_count) * kPatchDim, 0.0f);
        hunyuan_ocr::VisionRuntimeResult result;
        if (!runtime.run_dynamic_pixel_values(pixels,
                                              shape.height,
                                              shape.width,
                                              kMergeSize,
                                              &result,
                                              &error))
        {
            std::cerr << "loop " << loop << ": " << error << '\n';
            return 3;
        }

        const int patch_h = shape.height / kMergeSize;
        const int patch_w = shape.width / kMergeSize;
        const int expected_tokens = patch_h * (patch_w + 1) + 2;
        const size_t expected_values =
            static_cast<size_t>(expected_tokens) * kHiddenSize;
        if (result.patch_count != patch_count ||
            result.vision_token_count != expected_tokens ||
            result.feature_values != expected_values ||
            result.vision_features.size() != expected_values ||
            !features_are_finite(result.vision_features))
        {
            std::cerr << "dynamic vision output mismatch at loop " << loop << '\n';
            return 4;
        }
    }

    std::cout << "inference_loops=" << kInferenceLoops << '\n'
              << "shape_1_grid=" << shapes[0].height << 'x' << shapes[0].width << '\n'
              << "shape_2_grid=" << shapes[1].height << 'x' << shapes[1].width << '\n';
    return 0;
}
