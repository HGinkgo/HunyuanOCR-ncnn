#pragma once

#include <memory>
#include <string>
#include <vector>

#include <net.h>

namespace hunyuan_ocr {

class MappedModelFile;

namespace detail {

float bilinear_source_coordinate(int output_index, int input_size, int output_size);

} // namespace detail

struct VisionRuntimeResult {
    int patch_count = 0;
    int vision_token_count = 0;
    size_t feature_values = 0;
    std::vector<float> vision_features;
};

struct VisionRuntimeOptions {
    int num_threads = 0;
    bool use_vulkan = false;
    int vulkan_device = 0;
    bool mmap_weights = false;
};

bool vision_vulkan_compiled();

class VisionRuntime {
public:
    explicit VisionRuntime(int num_threads = 0);
    explicit VisionRuntime(const VisionRuntimeOptions& options);

    bool load_dynamic(const std::string& param_path,
                      const std::string& bin_path,
                      const std::string& pos_embed_path,
                      std::string* error);
    bool ready() const;
    size_t mapped_weight_bytes() const;
    int gelu_cpu_fallback_count() const;
    bool run_dynamic_pixel_values(const std::vector<float>& pixel_values,
                                  int grid_h,
                                  int grid_w,
                                  int merge_size,
                                  VisionRuntimeResult* result,
                                  std::string* error) const;

private:
    bool reset_net(std::string* error);
    bool load_net_files(const std::string& param_path,
                        const std::string& bin_path,
                        const char* model_label,
                        std::string* error);

    std::shared_ptr<MappedModelFile> vision_model_mapping_;
    std::unique_ptr<ncnn::Net> vision_net_;
    std::vector<float> pos_embed_base_;
    VisionRuntimeOptions options_;
    int gelu_cpu_fallback_count_ = 0;
    bool ready_ = false;
};

} // namespace hunyuan_ocr
