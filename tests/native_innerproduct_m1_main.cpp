#include "native_innerproduct_m1.h"

#include <platform.h>

#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Arguments {
    int input_size = 3584;
    int output_size = 1024;
    int warmup = 5;
    int repeat = 30;
    int device = 0;
};

bool parse_int(const char* text, int* value, bool allow_zero = false)
{
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || parsed < (allow_zero ? 0 : 1) ||
        parsed > std::numeric_limits<int>::max())
    {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool parse_arguments(int argc, char** argv, Arguments* arguments)
{
    for (int i = 1; i < argc; i += 2)
    {
        if (i + 1 >= argc)
        {
            return false;
        }
        int value = 0;
        const std::string option = argv[i];
        if (!parse_int(argv[i + 1], &value, option == "--device")) return false;
        if (option == "--input-size") arguments->input_size = value;
        else if (option == "--output-size") arguments->output_size = value;
        else if (option == "--warmup") arguments->warmup = value;
        else if (option == "--repeat") arguments->repeat = value;
        else if (option == "--device") arguments->device = value;
        else return false;
    }
    return arguments->device >= 0;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5
                                  : values[middle];
}

double benchmark(const hunyuan_ocr::diag::NativeLinearRuntime& runtime,
                 const std::vector<float>& input,
                 int warmup,
                 int repeat,
                 std::string* error)
{
    std::vector<float> output;
    for (int i = 0; i < warmup; ++i)
    {
        if (!runtime.run(input, &output, error)) return -1.0;
    }
    std::vector<double> samples;
    samples.reserve(repeat);
    for (int i = 0; i < repeat; ++i)
    {
        const auto begin = std::chrono::steady_clock::now();
        if (!runtime.run(input, &output, error)) return -1.0;
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return median(std::move(samples));
}

float max_abs_error(const std::vector<float>& left, const std::vector<float>& right)
{
    float result = 0.f;
    for (size_t i = 0; i < left.size(); ++i)
    {
        result = std::max(result, std::fabs(left[i] - right[i]));
    }
    return result;
}

double cosine_similarity(const std::vector<float>& left, const std::vector<float>& right)
{
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (size_t i = 0; i < left.size(); ++i)
    {
        dot += static_cast<double>(left[i]) * right[i];
        left_norm += static_cast<double>(left[i]) * left[i];
        right_norm += static_cast<double>(right[i]) * right[i];
    }
    return dot / std::sqrt(left_norm * right_norm);
}

int run(const Arguments& arguments)
{
#if !NCNN_VULKAN
    (void)arguments;
    std::cerr << "ncnn was built without Vulkan support\n";
    return 2;
#else
    if (ncnn::create_gpu_instance() != 0)
    {
        std::cerr << "failed to initialize ncnn Vulkan\n";
        return 2;
    }
    struct GpuGuard {
        ~GpuGuard() { ncnn::destroy_gpu_instance(); }
    } guard;
    if (arguments.device >= ncnn::get_gpu_count())
    {
        std::cerr << "requested Vulkan device is unavailable\n";
        return 2;
    }

    std::vector<float> weights(static_cast<size_t>(arguments.input_size) *
                               arguments.output_size);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = 0.025f * std::sin(static_cast<float>(i % 65521) * 0.0137f) +
                     0.01f * std::cos(static_cast<float>(i % 32749) * 0.0071f);
    }
    std::vector<float> input(arguments.input_size);
    float input_absmax = 0.f;
    for (int i = 0; i < arguments.input_size; ++i)
    {
        input[i] = 2.f * std::sin(static_cast<float>(i) * 0.031f) +
                   0.5f * std::cos(static_cast<float>(i) * 0.017f);
        input_absmax = std::max(input_absmax, std::fabs(input[i]));
    }
    const float activation_scale = 127.f / input_absmax;

    hunyuan_ocr::diag::NativeLinearRuntime gemm;
    hunyuan_ocr::diag::NativeLinearRuntime innerproduct;
    hunyuan_ocr::diag::NativeLinearRuntime innerproduct_int8;
    std::string error;
    if (!gemm.load(hunyuan_ocr::diag::NativeLinearPath::GemmFp32,
                   arguments.input_size, arguments.output_size, weights,
                   activation_scale, true, arguments.device, &error) ||
        !innerproduct.load(hunyuan_ocr::diag::NativeLinearPath::InnerProductFp32,
                           arguments.input_size, arguments.output_size, weights,
                           activation_scale, true, arguments.device, &error) ||
        !innerproduct_int8.load(hunyuan_ocr::diag::NativeLinearPath::InnerProductInt8,
                                arguments.input_size, arguments.output_size, weights,
                                activation_scale, true, arguments.device, &error))
    {
        std::cerr << error << '\n';
        return 3;
    }

    std::vector<float> gemm_output;
    std::vector<float> innerproduct_output;
    std::vector<float> int8_output;
    if (!gemm.run(input, &gemm_output, &error) ||
        !innerproduct.run(input, &innerproduct_output, &error) ||
        !innerproduct_int8.run(input, &int8_output, &error))
    {
        std::cerr << error << '\n';
        return 4;
    }

    const double gemm_ms = benchmark(gemm, input, arguments.warmup, arguments.repeat, &error);
    const double innerproduct_ms =
        benchmark(innerproduct, input, arguments.warmup, arguments.repeat, &error);
    const double int8_ms =
        benchmark(innerproduct_int8, input, arguments.warmup, arguments.repeat, &error);
    if (gemm_ms <= 0.0 || innerproduct_ms <= 0.0 || int8_ms <= 0.0)
    {
        std::cerr << error << '\n';
        return 5;
    }

    std::cout << "input_size=" << arguments.input_size << '\n'
              << "output_size=" << arguments.output_size << '\n'
              << "warmup=" << arguments.warmup << '\n'
              << "repeat=" << arguments.repeat << '\n'
              << "activation_scale=" << activation_scale << '\n'
              << "gemm_fp32_median_ms=" << gemm_ms << '\n'
              << "innerproduct_fp32_median_ms=" << innerproduct_ms << '\n'
              << "innerproduct_fp32_speedup=" << gemm_ms / innerproduct_ms << '\n'
              << "innerproduct_int8_median_ms=" << int8_ms << '\n'
              << "innerproduct_int8_speedup=" << gemm_ms / int8_ms << '\n'
              << "fp32_max_abs_error="
              << max_abs_error(gemm_output, innerproduct_output) << '\n'
              << "int8_max_abs_error=" << max_abs_error(gemm_output, int8_output) << '\n'
              << "int8_cosine_similarity=" << cosine_similarity(gemm_output, int8_output)
              << '\n';
    return 0;
#endif
}

} // namespace

int main(int argc, char** argv)
{
    Arguments arguments;
    if (!parse_arguments(argc, argv, &arguments))
    {
        std::cerr << "usage: " << argv[0]
                  << " [--input-size K] [--output-size N] [--warmup N]"
                     " [--repeat N] [--device ZERO_BASED_INDEX]\n";
        return 1;
    }
    return run(arguments);
}
