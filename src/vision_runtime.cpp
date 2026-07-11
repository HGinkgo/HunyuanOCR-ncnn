#include "hunyuan_ocr/vision_runtime.h"

#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/utf8.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

namespace hunyuan_ocr {

namespace detail {

float bilinear_source_coordinate(int output_index, int input_size, int output_size)
{
    return (static_cast<float>(output_index) + 0.5f) *
               static_cast<float>(input_size) / static_cast<float>(output_size) -
           0.5f;
}

} // namespace detail

namespace {

constexpr int kPatchDim = 768;
constexpr int kHiddenSize = 1024;
constexpr int kPatchSize = 16;
constexpr int kChannelCount = 3;
constexpr int kVisionHiddenSize = 1152;
constexpr int kPositionEdge = 128;

template <typename T>
bool read_binary_vector(const std::filesystem::path& path, size_t expected_count, std::vector<T>* values, std::string* error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        if (error) *error = "failed to open: " + path_to_utf8(path);
        return false;
    }
    values->assign(expected_count, T{});
    file.read(reinterpret_cast<char*>(values->data()), static_cast<std::streamsize>(expected_count * sizeof(T)));
    if (file.gcount() != static_cast<std::streamsize>(expected_count * sizeof(T)))
    {
        if (error) *error = "short read: " + path_to_utf8(path);
        return false;
    }
    char extra = 0;
    if (file.read(&extra, 1))
    {
        if (error) *error = "unexpected extra bytes: " + path_to_utf8(path);
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
        if (error) *error = "failed to open vision fixture meta: " + path_to_utf8(path);
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

bool pixel_values_to_chw(const std::vector<float>& pixel_values,
                         int grid_h,
                         int grid_w,
                         int merge_size,
                         std::vector<float>* chw,
                         std::string* error)
{
    if (chw == nullptr)
    {
        if (error) *error = "chw pointer is null";
        return false;
    }
    if (grid_h <= 0 || grid_w <= 0 || merge_size <= 0)
    {
        if (error) *error = "grid_h, grid_w, and merge_size must be positive";
        return false;
    }
    if (grid_h % merge_size != 0 || grid_w % merge_size != 0)
    {
        if (error) *error = "grid_h and grid_w must be divisible by merge_size";
        return false;
    }

    const int patch_count = grid_h * grid_w;
    const size_t expected = static_cast<size_t>(patch_count) * kPatchDim;
    if (pixel_values.size() != expected)
    {
        if (error) {
            *error = "pixel_values size mismatch for dynamic vision: got " + std::to_string(pixel_values.size()) +
                     ", expected " + std::to_string(expected);
        }
        return false;
    }

    const int height = grid_h * kPatchSize;
    const int width = grid_w * kPatchSize;
    chw->assign(static_cast<size_t>(kChannelCount) * height * width, 0.0f);

    size_t in_index = 0;
    for (int gh_block = 0; gh_block < grid_h / merge_size; ++gh_block)
    {
        for (int merge_h = 0; merge_h < merge_size; ++merge_h)
        {
            for (int gw_block = 0; gw_block < grid_w / merge_size; ++gw_block)
            {
                for (int merge_w = 0; merge_w < merge_size; ++merge_w)
                {
                    const int patch_y0 = (gh_block * merge_size + merge_h) * kPatchSize;
                    const int patch_x0 = (gw_block * merge_size + merge_w) * kPatchSize;
                    for (int channel = 0; channel < kChannelCount; ++channel)
                    {
                        for (int y = 0; y < kPatchSize; ++y)
                        {
                            for (int x = 0; x < kPatchSize; ++x)
                            {
                                const size_t out_index =
                                    (static_cast<size_t>(channel) * height + patch_y0 + y) * width + patch_x0 + x;
                                (*chw)[out_index] = pixel_values[in_index++];
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

void interpolate_pos_embed(const std::vector<float>& base,
                           int grid_h,
                           int grid_w,
                           std::vector<float>* out)
{
    out->assign(static_cast<size_t>(kVisionHiddenSize) * grid_h * grid_w, 0.0f);
    for (int y = 0; y < grid_h; ++y)
    {
        const float in_y = detail::bilinear_source_coordinate(y, kPositionEdge, grid_h);
        int y0 = static_cast<int>(std::floor(in_y));
        float wy = in_y - static_cast<float>(y0);
        if (y0 < 0)
        {
            y0 = 0;
            wy = 0.0f;
        }
        int y1 = y0 + 1;
        if (y1 >= kPositionEdge)
        {
            y1 = kPositionEdge - 1;
            y0 = y1;
            wy = 0.0f;
        }

        for (int x = 0; x < grid_w; ++x)
        {
            const float in_x = detail::bilinear_source_coordinate(x, kPositionEdge, grid_w);
            int x0 = static_cast<int>(std::floor(in_x));
            float wx = in_x - static_cast<float>(x0);
            if (x0 < 0)
            {
                x0 = 0;
                wx = 0.0f;
            }
            int x1 = x0 + 1;
            if (x1 >= kPositionEdge)
            {
                x1 = kPositionEdge - 1;
                x0 = x1;
                wx = 0.0f;
            }

            const float w00 = (1.0f - wy) * (1.0f - wx);
            const float w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx);
            const float w11 = wy * wx;

            for (int channel = 0; channel < kVisionHiddenSize; ++channel)
            {
                const size_t base_channel = static_cast<size_t>(channel) * kPositionEdge * kPositionEdge;
                const float v00 = base[base_channel + static_cast<size_t>(y0) * kPositionEdge + x0];
                const float v01 = base[base_channel + static_cast<size_t>(y0) * kPositionEdge + x1];
                const float v10 = base[base_channel + static_cast<size_t>(y1) * kPositionEdge + x0];
                const float v11 = base[base_channel + static_cast<size_t>(y1) * kPositionEdge + x1];
                (*out)[(static_cast<size_t>(channel) * grid_h + y) * grid_w + x] =
                    v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
            }
        }
    }
}

} // namespace

bool VisionRuntimeResult::matches_expected(float tolerance) const
{
    return has_expected_features && max_abs_diff_expected <= tolerance;
}

VisionRuntime::VisionRuntime(int num_threads)
    : vision_net_(new ncnn::Net),
      num_threads_(num_threads)
{
}

bool VisionRuntime::load(const std::string& param_path, const std::string& bin_path, std::string* error)
{
    const std::filesystem::path native_param_path = path_from_utf8(param_path);
    const std::filesystem::path native_bin_path = path_from_utf8(bin_path);
    ready_ = false;
    dynamic_ready_ = false;
    pos_embed_base_.clear();
    vision_net_.reset(new ncnn::Net);
    vision_net_->opt = make_fp32_ncnn_option(num_threads_);
    vision_net_->opt.use_packing_layout = false;

    if (vision_net_->load_param(native_param_path.c_str()) != 0)
    {
        if (error) *error = "failed to load vision param: " + param_path;
        return false;
    }
    if (vision_net_->load_model(native_bin_path.c_str()) != 0)
    {
        if (error) *error = "failed to load vision bin: " + bin_path;
        return false;
    }
    ready_ = true;
    return true;
}

bool VisionRuntime::load_dynamic(const std::string& param_path,
                                 const std::string& bin_path,
                                 const std::string& pos_embed_path,
                                 std::string* error)
{
    const std::filesystem::path native_param_path = path_from_utf8(param_path);
    const std::filesystem::path native_bin_path = path_from_utf8(bin_path);
    const std::filesystem::path native_pos_embed_path = path_from_utf8(pos_embed_path);
    ready_ = false;
    dynamic_ready_ = false;
    pos_embed_base_.clear();
    vision_net_.reset(new ncnn::Net);
    vision_net_->opt = make_fp32_ncnn_option(num_threads_);
    vision_net_->opt.use_packing_layout = false;

    if (vision_net_->load_param(native_param_path.c_str()) != 0)
    {
        if (error) *error = "failed to load dynamic vision param: " + param_path;
        return false;
    }
    if (vision_net_->load_model(native_bin_path.c_str()) != 0)
    {
        if (error) *error = "failed to load dynamic vision bin: " + bin_path;
        return false;
    }
    if (!read_binary_vector(native_pos_embed_path,
                            static_cast<size_t>(kVisionHiddenSize) * kPositionEdge * kPositionEdge,
                            &pos_embed_base_,
                            error))
    {
        return false;
    }

    ready_ = true;
    dynamic_ready_ = true;
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

bool VisionRuntime::run_dynamic_pixel_values(const std::vector<float>& pixel_values,
                                             int grid_h,
                                             int grid_w,
                                             int merge_size,
                                             VisionRuntimeResult* result,
                                             std::string* error) const
{
    if (!ready_ || !dynamic_ready_)
    {
        if (error) *error = "dynamic vision runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }
    if (grid_h <= 0 || grid_w <= 0 || merge_size <= 0)
    {
        if (error) *error = "grid_h, grid_w, and merge_size must be positive";
        return false;
    }

    std::vector<float> image_chw;
    if (!pixel_values_to_chw(pixel_values, grid_h, grid_w, merge_size, &image_chw, error))
    {
        return false;
    }

    std::vector<float> pos_embed;
    interpolate_pos_embed(pos_embed_base_, grid_h, grid_w, &pos_embed);

    const int image_w = grid_w * kPatchSize;
    const int image_h = grid_h * kPatchSize;
    ncnn::Mat image_input(image_w, image_h, kChannelCount, image_chw.data());
    image_input = image_input.clone();

    ncnn::Mat pos_input(grid_w, grid_h, kVisionHiddenSize, pos_embed.data());
    pos_input = pos_input.clone();

    ncnn::Mat output;
    ncnn::Extractor ex = vision_net_->create_extractor();
    if (ex.input("in0", image_input) != 0)
    {
        if (error) *error = "dynamic vision input in0 failed";
        return false;
    }
    if (ex.input("in1", pos_input) != 0)
    {
        if (error) *error = "dynamic vision input in1 failed";
        return false;
    }
    if (ex.extract("out0", output) != 0)
    {
        if (error) *error = "dynamic vision extract out0 failed";
        return false;
    }

    const int patch_h = grid_h / merge_size;
    const int patch_w = grid_w / merge_size;
    const int vision_token_count = patch_h * (patch_w + 1) + 2;
    const size_t expected_values = static_cast<size_t>(vision_token_count) * kHiddenSize;
    const size_t actual_values = logical_value_count(output);
    if (actual_values != expected_values)
    {
        if (error) {
            *error = "dynamic vision output size mismatch: got " + std::to_string(actual_values) +
                     ", expected " + std::to_string(expected_values);
        }
        return false;
    }

    VisionRuntimeResult local;
    local.patch_count = grid_h * grid_w;
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

    const std::filesystem::path root = path_from_utf8(fixture_dir);
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
