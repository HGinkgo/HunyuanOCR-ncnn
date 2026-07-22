#include "native_innerproduct_m1.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

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

bool test_fp32_innerproduct_matches_gemm()
{
    const int input_size = 4;
    const int output_size = 3;
    const std::vector<float> weights = {
        1.f, 0.f, -1.f, 0.5f,
        -0.5f, 1.f, 0.25f, -1.f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };
    const std::vector<float> input = {1.f, -1.f, 0.5f, -0.5f};

    hunyuan_ocr::diag::NativeLinearRuntime gemm;
    hunyuan_ocr::diag::NativeLinearRuntime innerproduct;
    std::string error;
    if (!expect(gemm.load(hunyuan_ocr::diag::NativeLinearPath::GemmFp32,
                          input_size,
                          output_size,
                          weights,
                          1.f,
                          false,
                          0,
                          &error),
                "failed to load FP32 Gemm") ||
        !expect(innerproduct.load(hunyuan_ocr::diag::NativeLinearPath::InnerProductFp32,
                                  input_size,
                                  output_size,
                                  weights,
                                  1.f,
                                  false,
                                  0,
                                  &error),
                "failed to load FP32 InnerProduct"))
    {
        std::cerr << error << '\n';
        return false;
    }

    std::vector<float> gemm_output;
    std::vector<float> innerproduct_output;
    if (!expect(gemm.run(input, &gemm_output, &error), "FP32 Gemm inference failed") ||
        !expect(innerproduct.run(input, &innerproduct_output, &error),
                "FP32 InnerProduct inference failed"))
    {
        std::cerr << error << '\n';
        return false;
    }

    if (!expect(gemm_output.size() == static_cast<size_t>(output_size) &&
                    innerproduct_output.size() == gemm_output.size(),
                "FP32 output shape mismatch"))
    {
        return false;
    }
    for (size_t i = 0; i < gemm_output.size(); ++i)
    {
        if (!expect(std::fabs(gemm_output[i] - innerproduct_output[i]) < 1e-6f,
                    "FP32 InnerProduct differs from Gemm"))
        {
            return false;
        }
    }
    return true;
}

bool test_int8_innerproduct_uses_native_quantized_path()
{
    const int input_size = 4;
    const int output_size = 3;
    const std::vector<float> weights = {
        1.f, 0.f, -1.f, 0.5f,
        -0.5f, 1.f, 0.25f, -1.f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };
    const std::vector<float> input = {1.f, -1.f, 0.5f, -0.5f};

    hunyuan_ocr::diag::NativeLinearRuntime fp32;
    hunyuan_ocr::diag::NativeLinearRuntime int8;
    std::string error;
    if (!fp32.load(hunyuan_ocr::diag::NativeLinearPath::InnerProductFp32,
                   input_size,
                   output_size,
                   weights,
                   1.f,
                   false,
                   0,
                   &error) ||
        !int8.load(hunyuan_ocr::diag::NativeLinearPath::InnerProductInt8,
                   input_size,
                   output_size,
                   weights,
                   127.f,
                   false,
                   0,
                   &error))
    {
        std::cerr << error << '\n';
        return false;
    }

    std::vector<float> fp32_output;
    std::vector<float> int8_output;
    if (!fp32.run(input, &fp32_output, &error) || !int8.run(input, &int8_output, &error))
    {
        std::cerr << error << '\n';
        return false;
    }

    float max_abs_error = 0.f;
    for (size_t i = 0; i < fp32_output.size(); ++i)
    {
        max_abs_error = std::max(max_abs_error, std::fabs(fp32_output[i] - int8_output[i]));
    }
    return expect(max_abs_error < 0.02f, "native INT8 InnerProduct error is too large");
}

} // namespace

int main()
{
    if (!test_fp32_innerproduct_matches_gemm() ||
        !test_int8_innerproduct_uses_native_quantized_path())
    {
        return 1;
    }
    return 0;
}
