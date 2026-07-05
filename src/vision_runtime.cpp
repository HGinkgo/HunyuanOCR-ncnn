#include "hunyuan_ocr/vision_runtime.h"

#include "hunyuan_ocr/hunyuan_ocr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

namespace hunyuan_ocr {
namespace {

constexpr int kPatchDim = 768;
constexpr int kHiddenSize = 1024;

template <typename T>
bool read_binary_vector(const std::filesystem::path& path, size_t expected_count, std::vector<T>* values, std::string* error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        if (error) *error = "failed to open: " + path.string();
        return false;
    }
    values->assign(expected_count, T{});
    file.read(reinterpret_cast<char*>(values->data()), static_cast<std::streamsize>(expected_count * sizeof(T)));
    if (file.gcount() != static_cast<std::streamsize>(expected_count * sizeof(T)))
    {
        if (error) *error = "short read: " + path.string();
        return false;
    }
    char extra = 0;
    if (file.read(&extra, 1))
    {
        if (error) *error = "unexpected extra bytes: " + path.string();
        return false;
    }
    return true;
}

bool file_exists(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

bool parse_fixture_meta(const std::filesystem::path& path, int* patch_count, int* vision_token_count, std::string* error)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        if (error) *error = "failed to open vision fixture meta: " + path.string();
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        try
        {
            if (key == "patch_count") *patch_count = std::stoi(value);
            else if (key == "vision_token_count") *vision_token_count = std::stoi(value);
        }
        catch (const std::exception&)
        {
            if (error) *error = "invalid vision fixture meta value: " + line;
            return false;
        }
    }

    if (*patch_count <= 0 || *vision_token_count <= 0)
    {
        if (error) *error = "vision fixture meta must define positive patch_count and vision_token_count";
        return false;
    }
    return true;
}

size_t logical_value_count(const ncnn::Mat& mat)
{
    const int h = mat.h > 0 ? mat.h : 1;
    const int d = mat.d > 0 ? mat.d : 1;
    const int c = mat.c > 0 ? mat.c : 1;
    const int elempack = mat.elempack > 0 ? mat.elempack : 1;
    return static_cast<size_t>(mat.w) * static_cast<size_t>(h) *
           static_cast<size_t>(d) * static_cast<size_t>(c) *
           static_cast<size_t>(elempack);
}

} // namespace

bool VisionRuntimeResult::matches_expected(float tolerance) const
{
    return has_expected_features && max_abs_diff_expected <= tolerance;
}

VisionRuntime::VisionRuntime()
    : vision_net_(new ncnn::Net)
{
}

bool VisionRuntime::load(const std::string& param_path, const std::string& bin_path, std::string* error)
{
    ready_ = false;
    vision_net_.reset(new ncnn::Net);
    vision_net_->opt = make_fp32_ncnn_option();
    vision_net_->opt.use_packing_layout = false;

    if (vision_net_->load_param(param_path.c_str()) != 0)
    {
        if (error) *error = "failed to load vision param: " + param_path;
        return false;
    }
    if (vision_net_->load_model(bin_path.c_str()) != 0)
    {
        if (error) *error = "failed to load vision bin: " + bin_path;
        return false;
    }
    ready_ = true;
    return true;
}

bool VisionRuntime::ready() const
{
    return ready_;
}

bool VisionRuntime::run_pixel_values(const std::vector<float>& pixel_values,
                                     int patch_count,
                                     int vision_token_count,
                                     VisionRuntimeResult* result,
                                     std::string* error) const
{
    if (!ready_)
    {
        if (error) *error = "vision runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }
    if (patch_count <= 0 || vision_token_count <= 0)
    {
        if (error) *error = "patch_count and vision_token_count must be positive";
        return false;
    }
    if (pixel_values.size() != static_cast<size_t>(patch_count) * kPatchDim)
    {
        if (error) {
            *error = "pixel_values size mismatch: got " + std::to_string(pixel_values.size()) +
                     ", expected " + std::to_string(static_cast<size_t>(patch_count) * kPatchDim);
        }
        return false;
    }

    ncnn::Mat input(kPatchDim, patch_count, const_cast<float*>(pixel_values.data()));
    input = input.clone();

    ncnn::Mat output;
    ncnn::Extractor ex = vision_net_->create_extractor();
    if (ex.input("in0", input) != 0)
    {
        if (error) *error = "vision input failed";
        return false;
    }
    if (ex.extract("out0", output) != 0)
    {
        if (error) *error = "vision extract out0 failed";
        return false;
    }

    const size_t expected_values = static_cast<size_t>(vision_token_count) * kHiddenSize;
    const size_t actual_values = logical_value_count(output);
    if (actual_values != expected_values)
    {
        if (error) {
            *error = "vision output size mismatch: got " + std::to_string(actual_values) +
                     ", expected " + std::to_string(expected_values);
        }
        return false;
    }

    VisionRuntimeResult local;
    local.patch_count = patch_count;
    local.vision_token_count = vision_token_count;
    local.feature_values = actual_values;
    local.vision_features.assign(expected_values, 0.0f);

    if (output.h == vision_token_count && output.w == kHiddenSize)
    {
        for (int row_index = 0; row_index < vision_token_count; ++row_index)
        {
            const float* src = output.row(row_index);
            std::memcpy(local.vision_features.data() + static_cast<size_t>(row_index) * kHiddenSize,
                        src,
                        static_cast<size_t>(kHiddenSize) * sizeof(float));
        }
    }
    else
    {
        const float* src = output;
        std::memcpy(local.vision_features.data(), src, expected_values * sizeof(float));
    }

    *result = std::move(local);
    return true;
}

bool VisionRuntime::run_fixture(const std::string& fixture_dir, VisionRuntimeResult* result, std::string* error) const
{
    if (!ready_)
    {
        if (error) *error = "vision runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }

    const std::filesystem::path root(fixture_dir);
    int patch_count = 0;
    int vision_token_count = 0;
    if (!parse_fixture_meta(root / "meta.txt", &patch_count, &vision_token_count, error))
    {
        return false;
    }

    std::vector<float> pixel_values;
    if (!read_binary_vector(root / "pixel_values.f32",
                            static_cast<size_t>(patch_count) * kPatchDim,
                            &pixel_values,
                            error))
    {
        return false;
    }

    const size_t expected_values = static_cast<size_t>(vision_token_count) * kHiddenSize;
    VisionRuntimeResult local;
    if (!run_pixel_values(pixel_values, patch_count, vision_token_count, &local, error))
    {
        return false;
    }

    const std::filesystem::path expected_path = root / "expected_vision_features.f32";
    if (file_exists(expected_path))
    {
        std::vector<float> expected;
        if (!read_binary_vector(expected_path, expected_values, &expected, error))
        {
            return false;
        }
        double sum_abs = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected_values; ++i)
        {
            const float diff = std::fabs(local.vision_features[i] - expected[i]);
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
        }
        local.has_expected_features = true;
        local.max_abs_diff_expected = max_abs;
        local.mean_abs_diff_expected = static_cast<float>(sum_abs / static_cast<double>(expected_values));
    }

    *result = std::move(local);
    return true;
}

} // namespace hunyuan_ocr
