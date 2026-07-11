#include "hunyuan_ocr/image_preprocessor.h"
#include "hunyuan_ocr/utf8.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#ifdef _WIN32
#define STBI_WINDOWS_UTF8
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace hunyuan_ocr {
namespace {

bool file_exists(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

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

long long python_round_to_int(double value)
{
    const double floor_value = std::floor(value);
    const double diff = value - floor_value;
    if (diff < 0.5) return static_cast<long long>(floor_value);
    if (diff > 0.5) return static_cast<long long>(floor_value) + 1;
    const long long lower = static_cast<long long>(floor_value);
    return (lower % 2 == 0) ? lower : lower + 1;
}

struct FixtureMeta {
    int original_width = 0;
    int original_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int grid_t = 1;
    int grid_h = 0;
    int grid_w = 0;
    int patch_count = 0;
};

struct ResizeCoefficients {
    std::vector<int> bounds;
    std::vector<int> weights;
    int taps = 0;
};

constexpr int kPrecisionBits = 22;

double bicubic_filter(double value)
{
    if (value < 0.0) value = -value;
    if (value < 1.0)
    {
        return ((1.5 * value - 2.5) * value) * value + 1.0;
    }
    if (value < 2.0)
    {
        return (((-0.5 * value + 2.5) * value - 4.0) * value) + 2.0;
    }
    return 0.0;
}

ResizeCoefficients precompute_resize_coefficients(int input_size, int output_size)
{
    constexpr double kSupport = 2.0;
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    const double filterscale = std::max(1.0, scale);
    const double support = kSupport * filterscale;
    const double inv_filterscale = 1.0 / filterscale;
    const int taps = static_cast<int>(std::ceil(support)) * 2 + 1;

    ResizeCoefficients coeffs;
    coeffs.taps = taps;
    coeffs.bounds.assign(static_cast<size_t>(output_size) * 2, 0);
    coeffs.weights.assign(static_cast<size_t>(output_size) * taps, 0);

    for (int out_index = 0; out_index < output_size; ++out_index)
    {
        const double center = (static_cast<double>(out_index) + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > input_size) xmax = input_size;
        if (xmax < xmin) xmax = xmin;

        std::vector<double> weights(static_cast<size_t>(taps), 0.0);
        double total = 0.0;
        for (int in_index = xmin; in_index < xmax; ++in_index)
        {
            const double weight = bicubic_filter((static_cast<double>(in_index) - center + 0.5) * inv_filterscale);
            weights[static_cast<size_t>(in_index - xmin)] = weight;
            total += weight;
        }
        const size_t output_offset = static_cast<size_t>(out_index) * taps;
        for (int tap = 0; tap < xmax - xmin; ++tap)
        {
            double normalized = weights[static_cast<size_t>(tap)];
            if (total != 0.0)
            {
                normalized /= total;
            }
            coeffs.weights[output_offset + static_cast<size_t>(tap)] =
                normalized < 0.0
                    ? static_cast<int>(-0.5 + normalized * static_cast<double>(1 << kPrecisionBits))
                    : static_cast<int>(0.5 + normalized * static_cast<double>(1 << kPrecisionBits));
        }
        coeffs.bounds[static_cast<size_t>(out_index) * 2] = xmin;
        coeffs.bounds[static_cast<size_t>(out_index) * 2 + 1] = xmax - xmin;
    }

    return coeffs;
}

unsigned char clip_fixedpoint_to_u8(long long value)
{
    const long long scaled = value >> kPrecisionBits;
    if (scaled <= 0) return 0;
    if (scaled >= 255) return 255;
    return static_cast<unsigned char>(scaled);
}

bool parse_fixture_meta(const std::filesystem::path& path, FixtureMeta* meta, std::string* error)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        if (error) *error = "failed to open image preprocess fixture meta: " + path_to_utf8(path);
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
            if (key == "original_width") meta->original_width = std::stoi(value);
            else if (key == "original_height") meta->original_height = std::stoi(value);
            else if (key == "resized_width") meta->resized_width = std::stoi(value);
            else if (key == "resized_height") meta->resized_height = std::stoi(value);
            else if (key == "grid_t") meta->grid_t = std::stoi(value);
            else if (key == "grid_h") meta->grid_h = std::stoi(value);
            else if (key == "grid_w") meta->grid_w = std::stoi(value);
            else if (key == "patch_count") meta->patch_count = std::stoi(value);
        }
        catch (const std::exception&)
        {
            if (error) *error = "invalid image preprocess fixture meta value: " + line;
            return false;
        }
    }

    if (meta->resized_width <= 0 ||
        meta->resized_height <= 0 ||
        meta->grid_t <= 0 ||
        meta->grid_h <= 0 ||
        meta->grid_w <= 0 ||
        meta->patch_count <= 0)
    {
        if (error) *error = "image preprocess fixture meta must define positive resized size, grid, and patch_count";
        return false;
    }
    return true;
}

} // namespace

bool ImagePreprocessResult::matches_expected(float tolerance) const
{
    return has_expected_pixel_values && max_abs_diff_expected <= tolerance;
}

ImagePreprocessor::ImagePreprocessor(ImagePreprocessConfig config)
    : config_(config)
{
}

const ImagePreprocessConfig& ImagePreprocessor::config() const
{
    return config_;
}

void ImagePreprocessor::smart_resize(int height, int width, int* resized_height, int* resized_width) const
{
    const long long factor = static_cast<long long>(config_.patch_size) * config_.merge_size;
    const double aspect = static_cast<double>(std::max(height, width)) / static_cast<double>(std::min(height, width));
    if (aspect > 200.0)
    {
        if (resized_height) *resized_height = 0;
        if (resized_width) *resized_width = 0;
        return;
    }

    long long h_bar = python_round_to_int(static_cast<double>(height) / static_cast<double>(factor)) * factor;
    long long w_bar = python_round_to_int(static_cast<double>(width) / static_cast<double>(factor)) * factor;
    if (h_bar < factor) h_bar = factor;
    if (w_bar < factor) w_bar = factor;

    const double area = static_cast<double>(height) * static_cast<double>(width);
    if (h_bar * w_bar > static_cast<long long>(config_.max_pixels))
    {
        const double beta = std::sqrt(area / static_cast<double>(config_.max_pixels));
        h_bar = std::max(factor, static_cast<long long>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<long long>(std::floor(width / beta / factor)) * factor);
    }
    else if (h_bar * w_bar < static_cast<long long>(config_.min_pixels))
    {
        const double beta = std::sqrt(static_cast<double>(config_.min_pixels) / area);
        h_bar = static_cast<long long>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<long long>(std::ceil(width * beta / factor)) * factor;
    }

    if (resized_height) *resized_height = static_cast<int>(h_bar);
    if (resized_width) *resized_width = static_cast<int>(w_bar);
}

bool ImagePreprocessor::preprocess_resized_rgb(const std::vector<unsigned char>& rgb,
                                               int width,
                                               int height,
                                               ImagePreprocessResult* result,
                                               std::string* error) const
{
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }
    if (width <= 0 || height <= 0)
    {
        if (error) *error = "image size must be positive";
        return false;
    }
    if (rgb.size() != static_cast<size_t>(width) * height * 3)
    {
        if (error) {
            *error = "rgb byte count mismatch: got " + std::to_string(rgb.size()) +
                     ", expected " + std::to_string(static_cast<size_t>(width) * height * 3);
        }
        return false;
    }
    if (width % config_.patch_size != 0 || height % config_.patch_size != 0)
    {
        if (error) *error = "resized image dimensions must be divisible by patch_size";
        return false;
    }

    const int grid_h = height / config_.patch_size;
    const int grid_w = width / config_.patch_size;
    if (grid_h % config_.merge_size != 0 || grid_w % config_.merge_size != 0)
    {
        if (error) *error = "image grid dimensions must be divisible by merge_size";
        return false;
    }

    ImagePreprocessResult local;
    local.resized_width = width;
    local.resized_height = height;
    local.grid_t = 1;
    local.grid_h = grid_h;
    local.grid_w = grid_w;
    local.patch_count = grid_h * grid_w;
    local.pixel_value_count = static_cast<size_t>(local.patch_count) * 3 * config_.patch_size * config_.patch_size;
    local.pixel_values.assign(local.pixel_value_count, 0.0f);

    size_t out_index = 0;
    for (int gh_block = 0; gh_block < grid_h / config_.merge_size; ++gh_block)
    {
        for (int merge_h = 0; merge_h < config_.merge_size; ++merge_h)
        {
            for (int gw_block = 0; gw_block < grid_w / config_.merge_size; ++gw_block)
            {
                for (int merge_w = 0; merge_w < config_.merge_size; ++merge_w)
                {
                    const int patch_y0 = (gh_block * config_.merge_size + merge_h) * config_.patch_size;
                    const int patch_x0 = (gw_block * config_.merge_size + merge_w) * config_.patch_size;
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        for (int y = 0; y < config_.patch_size; ++y)
                        {
                            for (int x = 0; x < config_.patch_size; ++x)
                            {
                                const size_t src_index =
                                    (static_cast<size_t>(patch_y0 + y) * width + patch_x0 + x) * 3 + channel;
                                const float scaled = static_cast<float>(rgb[src_index]) / 255.0f;
                                local.pixel_values[out_index++] = (scaled - config_.mean[channel]) / config_.std[channel];
                            }
                        }
                    }
                }
            }
        }
    }

    *result = std::move(local);
    return true;
}

bool ImagePreprocessor::resize_rgb_bicubic(const std::vector<unsigned char>& rgb,
                                           int width,
                                           int height,
                                           int resized_width,
                                           int resized_height,
                                           std::vector<unsigned char>* resized_rgb,
                                           std::string* error) const
{
    if (resized_rgb == nullptr)
    {
        if (error) *error = "resized_rgb pointer is null";
        return false;
    }
    if (width <= 0 || height <= 0 || resized_width <= 0 || resized_height <= 0)
    {
        if (error) *error = "resize dimensions must be positive";
        return false;
    }
    if (rgb.size() != static_cast<size_t>(width) * height * 3)
    {
        if (error) {
            *error = "original rgb byte count mismatch: got " + std::to_string(rgb.size()) +
                     ", expected " + std::to_string(static_cast<size_t>(width) * height * 3);
        }
        return false;
    }

    if (width == resized_width && height == resized_height)
    {
        *resized_rgb = rgb;
        return true;
    }

    const ResizeCoefficients x_coeffs = precompute_resize_coefficients(width, resized_width);
    const ResizeCoefficients y_coeffs = precompute_resize_coefficients(height, resized_height);
    std::vector<unsigned char> horizontal(static_cast<size_t>(height) * resized_width * 3, 0);

    for (int y = 0; y < height; ++y)
    {
        for (int out_x = 0; out_x < resized_width; ++out_x)
        {
            const int xmin = x_coeffs.bounds[static_cast<size_t>(out_x) * 2];
            const int count = x_coeffs.bounds[static_cast<size_t>(out_x) * 2 + 1];
            const int* weights = x_coeffs.weights.data() + static_cast<size_t>(out_x) * x_coeffs.taps;
            for (int channel = 0; channel < 3; ++channel)
            {
                long long value = 1LL << (kPrecisionBits - 1);
                for (int tap = 0; tap < count; ++tap)
                {
                    const size_t src_index = (static_cast<size_t>(y) * width + xmin + tap) * 3 + channel;
                    value += static_cast<long long>(rgb[src_index]) * weights[tap];
                }
                horizontal[(static_cast<size_t>(y) * resized_width + out_x) * 3 + channel] =
                    clip_fixedpoint_to_u8(value);
            }
        }
    }

    resized_rgb->assign(static_cast<size_t>(resized_width) * resized_height * 3, 0);
    for (int out_y = 0; out_y < resized_height; ++out_y)
    {
        const int ymin = y_coeffs.bounds[static_cast<size_t>(out_y) * 2];
        const int count = y_coeffs.bounds[static_cast<size_t>(out_y) * 2 + 1];
        const int* weights = y_coeffs.weights.data() + static_cast<size_t>(out_y) * y_coeffs.taps;
        for (int x = 0; x < resized_width; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                long long value = 1LL << (kPrecisionBits - 1);
                for (int tap = 0; tap < count; ++tap)
                {
                    const size_t src_index = (static_cast<size_t>(ymin + tap) * resized_width + x) * 3 + channel;
                    value += horizontal[src_index] * weights[tap];
                }
                (*resized_rgb)[(static_cast<size_t>(out_y) * resized_width + x) * 3 + channel] =
                    clip_fixedpoint_to_u8(value);
            }
        }
    }

    return true;
}

bool ImagePreprocessor::preprocess_original_rgb(const std::vector<unsigned char>& rgb,
                                                int width,
                                                int height,
                                                ImagePreprocessResult* result,
                                                std::string* error) const
{
    int resized_height = 0;
    int resized_width = 0;
    smart_resize(height, width, &resized_height, &resized_width);
    if (resized_height <= 0 || resized_width <= 0)
    {
        if (error) *error = "smart_resize failed for image dimensions";
        return false;
    }

    std::vector<unsigned char> resized_rgb;
    if (!resize_rgb_bicubic(rgb, width, height, resized_width, resized_height, &resized_rgb, error))
    {
        return false;
    }

    if (!preprocess_resized_rgb(resized_rgb, resized_width, resized_height, result, error))
    {
        return false;
    }
    result->original_width = width;
    result->original_height = height;
    return true;
}

bool ImagePreprocessor::preprocess_image_file(const std::string& image_path,
                                              ImagePreprocessResult* result,
                                              std::string* error) const
{
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (decoded == nullptr)
    {
        if (error) {
            const char* reason = stbi_failure_reason();
            *error = "failed to decode image file: " + image_path;
            if (reason != nullptr)
            {
                *error += " (";
                *error += reason;
                *error += ")";
            }
        }
        return false;
    }

    const size_t byte_count = static_cast<size_t>(width) * height * 3;
    std::vector<unsigned char> rgb(decoded, decoded + byte_count);
    stbi_image_free(decoded);

    return preprocess_original_rgb(rgb, width, height, result, error);
}

bool ImagePreprocessor::run_fixture(const std::string& fixture_dir,
                                    ImagePreprocessResult* result,
                                    std::string* error) const
{
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }

    const std::filesystem::path root = path_from_utf8(fixture_dir);
    FixtureMeta meta;
    if (!parse_fixture_meta(root / "meta.txt", &meta, error))
    {
        return false;
    }

    std::vector<unsigned char> rgb;
    if (!read_binary_vector(root / "resized_rgb.u8",
                            static_cast<size_t>(meta.resized_width) * meta.resized_height * 3,
                            &rgb,
                            error))
    {
        return false;
    }

    ImagePreprocessResult local;
    if (!preprocess_resized_rgb(rgb, meta.resized_width, meta.resized_height, &local, error))
    {
        return false;
    }
    local.original_width = meta.original_width;
    local.original_height = meta.original_height;

    if (local.grid_t != meta.grid_t ||
        local.grid_h != meta.grid_h ||
        local.grid_w != meta.grid_w ||
        local.patch_count != meta.patch_count)
    {
        if (error) {
            std::ostringstream oss;
            oss << "fixture grid mismatch: got [" << local.grid_t << ", " << local.grid_h << ", " << local.grid_w
                << "] patches=" << local.patch_count << ", expected [" << meta.grid_t << ", " << meta.grid_h
                << ", " << meta.grid_w << "] patches=" << meta.patch_count;
            *error = oss.str();
        }
        return false;
    }

    const std::filesystem::path expected_path = root / "expected_pixel_values.f32";
    if (file_exists(expected_path))
    {
        std::vector<float> expected;
        if (!read_binary_vector(expected_path, local.pixel_value_count, &expected, error))
        {
            return false;
        }
        double sum_abs = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < local.pixel_value_count; ++i)
        {
            const float diff = std::fabs(local.pixel_values[i] - expected[i]);
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
        }
        local.has_expected_pixel_values = true;
        local.max_abs_diff_expected = max_abs;
        local.mean_abs_diff_expected = static_cast<float>(sum_abs / static_cast<double>(local.pixel_value_count));
    }

    *result = std::move(local);
    return true;
}

bool ImagePreprocessor::run_image_file_fixture(const std::string& fixture_dir,
                                               ImagePreprocessResult* result,
                                               std::string* error) const
{
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }

    const std::filesystem::path root = path_from_utf8(fixture_dir);
    FixtureMeta meta;
    if (!parse_fixture_meta(root / "meta.txt", &meta, error))
    {
        return false;
    }
    if (meta.original_width <= 0 || meta.original_height <= 0)
    {
        if (error) *error = "image file fixture meta must define positive original_width and original_height";
        return false;
    }

    std::vector<unsigned char> rgb;
    if (!read_binary_vector(root / "original_rgb.u8",
                            static_cast<size_t>(meta.original_width) * meta.original_height * 3,
                            &rgb,
                            error))
    {
        return false;
    }

    ImagePreprocessResult local;
    if (!preprocess_original_rgb(rgb, meta.original_width, meta.original_height, &local, error))
    {
        return false;
    }

    if (local.resized_width != meta.resized_width ||
        local.resized_height != meta.resized_height ||
        local.grid_t != meta.grid_t ||
        local.grid_h != meta.grid_h ||
        local.grid_w != meta.grid_w ||
        local.patch_count != meta.patch_count)
    {
        if (error) {
            std::ostringstream oss;
            oss << "image file fixture metadata mismatch: got resized=" << local.resized_width << "x"
                << local.resized_height << " grid=[" << local.grid_t << ", " << local.grid_h << ", "
                << local.grid_w << "] patches=" << local.patch_count << ", expected resized="
                << meta.resized_width << "x" << meta.resized_height << " grid=[" << meta.grid_t << ", "
                << meta.grid_h << ", " << meta.grid_w << "] patches=" << meta.patch_count;
            *error = oss.str();
        }
        return false;
    }

    const std::filesystem::path expected_path = root / "expected_pixel_values.f32";
    if (file_exists(expected_path))
    {
        std::vector<float> expected;
        if (!read_binary_vector(expected_path, local.pixel_value_count, &expected, error))
        {
            return false;
        }
        double sum_abs = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < local.pixel_value_count; ++i)
        {
            const float diff = std::fabs(local.pixel_values[i] - expected[i]);
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
        }
        local.has_expected_pixel_values = true;
        local.max_abs_diff_expected = max_abs;
        local.mean_abs_diff_expected = static_cast<float>(sum_abs / static_cast<double>(local.pixel_value_count));
    }

    *result = std::move(local);
    return true;
}

} // namespace hunyuan_ocr
