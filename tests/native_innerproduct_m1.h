#pragma once

#include <memory>
#include <string>
#include <vector>

namespace hunyuan_ocr::diag {

enum class NativeLinearPath {
    GemmFp32,
    InnerProductFp32,
    InnerProductInt8,
};

class NativeLinearRuntime {
public:
    NativeLinearRuntime();
    ~NativeLinearRuntime();

    NativeLinearRuntime(const NativeLinearRuntime&) = delete;
    NativeLinearRuntime& operator=(const NativeLinearRuntime&) = delete;

    bool load(NativeLinearPath path,
              int input_size,
              int output_size,
              const std::vector<float>& weights,
              float activation_scale,
              bool use_vulkan,
              int device,
              std::string* error);
    bool run(const std::vector<float>& input,
             std::vector<float>* output,
             std::string* error) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hunyuan_ocr::diag
