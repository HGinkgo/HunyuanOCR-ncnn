#pragma once

#include <memory>
#include <string>
#include <vector>

#include <net.h>

namespace hunyuan_ocr {

struct VisionRuntimeResult {
    int patch_count = 0;
    int vision_token_count = 0;
    size_t feature_values = 0;
    std::vector<float> vision_features;
    bool has_expected_features = false;
    float max_abs_diff_expected = 0.0f;
    float mean_abs_diff_expected = 0.0f;

    bool matches_expected(float tolerance) const;
};

class VisionRuntime {
public:
    VisionRuntime();

    bool load(const std::string& param_path, const std::string& bin_path, std::string* error);
    bool ready() const;
    bool run_pixel_values(const std::vector<float>& pixel_values,
                          int patch_count,
                          int vision_token_count,
                          VisionRuntimeResult* result,
                          std::string* error) const;
    bool run_fixture(const std::string& fixture_dir, VisionRuntimeResult* result, std::string* error) const;

private:
    std::unique_ptr<ncnn::Net> vision_net_;
    bool ready_ = false;
};

} // namespace hunyuan_ocr
