#pragma once

#include <string>
#include <vector>

namespace hunyuan_ocr {

struct ImagePreprocessConfig {
    int min_pixels = 262144;
    int max_pixels = 524288;
    int patch_size = 16;
    int merge_size = 2;
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct ImagePreprocessResult {
    int original_width = 0;
    int original_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int grid_t = 1;
    int grid_h = 0;
    int grid_w = 0;
    int patch_count = 0;
    size_t pixel_value_count = 0;
    std::vector<float> pixel_values;
    bool has_expected_pixel_values = false;
    float max_abs_diff_expected = 0.0f;
    float mean_abs_diff_expected = 0.0f;

    bool matches_expected(float tolerance) const;
};

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(ImagePreprocessConfig config = ImagePreprocessConfig());

    const ImagePreprocessConfig& config() const;
    void smart_resize(int height, int width, int* resized_height, int* resized_width) const;
    bool resize_rgb_bicubic(const std::vector<unsigned char>& rgb,
                            int width,
                            int height,
                            int resized_width,
                            int resized_height,
                            std::vector<unsigned char>* resized_rgb,
                            std::string* error) const;
    bool preprocess_original_rgb(const std::vector<unsigned char>& rgb,
                                 int width,
                                 int height,
                                 ImagePreprocessResult* result,
                                 std::string* error) const;
    bool preprocess_image_file(const std::string& image_path,
                               ImagePreprocessResult* result,
                               std::string* error) const;
    bool preprocess_resized_rgb(const std::vector<unsigned char>& rgb,
                                int width,
                                int height,
                                ImagePreprocessResult* result,
                                std::string* error) const;
    bool run_fixture(const std::string& fixture_dir,
                     ImagePreprocessResult* result,
                     std::string* error) const;
    bool run_image_file_fixture(const std::string& fixture_dir,
                                ImagePreprocessResult* result,
                                std::string* error) const;

private:
    ImagePreprocessConfig config_;
};

} // namespace hunyuan_ocr
