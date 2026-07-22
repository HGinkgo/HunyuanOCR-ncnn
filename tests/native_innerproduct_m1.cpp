#include "native_innerproduct_m1.h"

#include <datareader.h>
#include <net.h>

#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace hunyuan_ocr::diag {
namespace {

class BoundedDataReader final : public ncnn::DataReader {
public:
    explicit BoundedDataReader(const std::vector<unsigned char>& bytes)
        : bytes_(bytes)
    {
    }

    size_t read(void* buffer, size_t size) const override
    {
        if (!buffer || offset_ > bytes_.size() || size > bytes_.size() - offset_)
        {
            return 0;
        }
        std::memcpy(buffer, bytes_.data() + offset_, size);
        offset_ += size;
        return size;
    }

    size_t reference(size_t size, const void** buffer) const override
    {
        if (!buffer)
        {
            return 0;
        }
        *buffer = nullptr;
        if (offset_ > bytes_.size() || size > bytes_.size() - offset_)
        {
            return 0;
        }
        *buffer = bytes_.data() + offset_;
        offset_ += size;
        return size;
    }

private:
    const std::vector<unsigned char>& bytes_;
    mutable size_t offset_ = 0;
};

template <typename T>
void append_value(std::vector<unsigned char>* bytes, const T& value)
{
    const auto* begin = reinterpret_cast<const unsigned char*>(&value);
    bytes->insert(bytes->end(), begin, begin + sizeof(T));
}

template <typename T>
void append_values(std::vector<unsigned char>* bytes, const std::vector<T>& values)
{
    const auto* begin = reinterpret_cast<const unsigned char*>(values.data());
    bytes->insert(bytes->end(), begin, begin + values.size() * sizeof(T));
}

std::vector<unsigned char> make_fp32_model(const std::vector<float>& weights)
{
    std::vector<unsigned char> model;
    model.reserve(sizeof(uint32_t) + weights.size() * sizeof(float));
    append_value(&model, uint32_t{0});
    append_values(&model, weights);
    return model;
}

std::vector<unsigned char> make_int8_model(const std::vector<float>& weights,
                                           int input_size,
                                           int output_size,
                                           float activation_scale)
{
    std::vector<signed char> quantized(weights.size());
    std::vector<float> weight_scales(output_size);
    for (int row = 0; row < output_size; ++row)
    {
        const float* row_weights = weights.data() + static_cast<size_t>(row) * input_size;
        float absmax = 0.f;
        for (int col = 0; col < input_size; ++col)
        {
            absmax = std::max(absmax, std::fabs(row_weights[col]));
        }
        const float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        weight_scales[row] = scale;
        for (int col = 0; col < input_size; ++col)
        {
            const float scaled = row_weights[col] * scale;
            const int rounded = scaled >= 0.f ? static_cast<int>(scaled + 0.5f)
                                               : static_cast<int>(scaled - 0.5f);
            quantized[static_cast<size_t>(row) * input_size + col] =
                static_cast<signed char>(std::clamp(rounded, -127, 127));
        }
    }

    const size_t aligned_weight_size = (quantized.size() + 3u) & ~size_t{3u};
    std::vector<unsigned char> model;
    model.reserve(sizeof(uint32_t) + aligned_weight_size +
                  weight_scales.size() * sizeof(float) + sizeof(float));
    append_value(&model, uint32_t{0x000D4B38});
    const auto* quantized_bytes = reinterpret_cast<const unsigned char*>(quantized.data());
    model.insert(model.end(), quantized_bytes, quantized_bytes + quantized.size());
    model.resize(sizeof(uint32_t) + aligned_weight_size, 0);
    append_values(&model, weight_scales);
    append_value(&model, activation_scale);
    return model;
}

std::string make_param(NativeLinearPath path, int input_size, int output_size)
{
    std::ostringstream param;
    param << "7767517\n2 2\nInput in0 0 1 in0\n";
    if (path == NativeLinearPath::GemmFp32)
    {
        param << "Gemm linear 1 1 in0 out0"
              << " 10=-1 2=0 3=1 4=0 5=1 6=1 7=0"
              << " 8=" << output_size << " 9=" << input_size << '\n';
    }
    else
    {
        const int int8_scale_term = path == NativeLinearPath::InnerProductInt8 ? 1 : 0;
        param << "InnerProduct linear 1 1 in0 out0"
              << " 0=" << output_size << " 1=0"
              << " 2=" << static_cast<long long>(input_size) * output_size
              << " 8=" << int8_scale_term << '\n';
    }
    return param.str();
}

void set_error(std::string* error, const std::string& message)
{
    if (error)
    {
        *error = message;
    }
}

} // namespace

class NativeLinearRuntime::Impl {
public:
    NativeLinearPath path = NativeLinearPath::GemmFp32;
    int input_size = 0;
    int output_size = 0;
    bool loaded = false;
    std::vector<unsigned char> model;
    ncnn::Net net;
};

NativeLinearRuntime::NativeLinearRuntime()
    : impl_(std::make_unique<Impl>())
{
}

NativeLinearRuntime::~NativeLinearRuntime() = default;

bool NativeLinearRuntime::load(NativeLinearPath path,
                               int input_size,
                               int output_size,
                               const std::vector<float>& weights,
                               float activation_scale,
                               bool use_vulkan,
                               int device,
                               std::string* error)
{
    if (input_size <= 0 || output_size <= 0 ||
        weights.size() != static_cast<size_t>(input_size) * output_size)
    {
        set_error(error, "invalid linear dimensions or weight count");
        return false;
    }
    if (path == NativeLinearPath::InnerProductInt8 &&
        (!std::isfinite(activation_scale) || activation_scale <= 0.f))
    {
        set_error(error, "INT8 activation scale must be finite and positive");
        return false;
    }

    impl_ = std::make_unique<Impl>();
    impl_->path = path;
    impl_->input_size = input_size;
    impl_->output_size = output_size;
    impl_->net.opt.num_threads = 1;
    impl_->net.opt.use_local_pool_allocator = false;
    impl_->net.opt.use_packing_layout = true;
    impl_->net.opt.use_fp16_packed = false;
    impl_->net.opt.use_fp16_storage = false;
    impl_->net.opt.use_fp16_arithmetic = false;
    impl_->net.opt.use_int8_inference = path == NativeLinearPath::InnerProductInt8;
    impl_->net.opt.use_int8_storage = path == NativeLinearPath::InnerProductInt8;
    impl_->net.opt.use_int8_packed = path == NativeLinearPath::InnerProductInt8;

    if (use_vulkan)
    {
#if NCNN_VULKAN
        if (device < 0 || device >= ncnn::get_gpu_count())
        {
            set_error(error, "Vulkan device is unavailable; initialize ncnn GPU first");
            return false;
        }
        impl_->net.opt.use_vulkan_compute = true;
        impl_->net.set_vulkan_device(device);
#else
        (void)device;
        set_error(error, "ncnn was built without Vulkan support");
        return false;
#endif
    }

    const std::string param = make_param(path, input_size, output_size);
    if (impl_->net.load_param_mem(param.c_str()) != 0)
    {
        set_error(error, "failed to load generated ncnn param");
        return false;
    }

    impl_->model = path == NativeLinearPath::InnerProductInt8
                       ? make_int8_model(weights, input_size, output_size, activation_scale)
                       : make_fp32_model(weights);
    BoundedDataReader reader(impl_->model);
    if (impl_->net.load_model(reader) != 0)
    {
        set_error(error, "failed to load generated ncnn model");
        return false;
    }
    impl_->loaded = true;
    return true;
}

bool NativeLinearRuntime::run(const std::vector<float>& input,
                              std::vector<float>* output,
                              std::string* error) const
{
    if (!output || !impl_->loaded || input.size() != static_cast<size_t>(impl_->input_size))
    {
        set_error(error, "runtime is not loaded or input shape is invalid");
        return false;
    }

    ncnn::Mat input_mat(impl_->input_size, 1);
    if (input_mat.empty())
    {
        set_error(error, "failed to allocate input tensor");
        return false;
    }
    std::memcpy(input_mat.data, input.data(), input.size() * sizeof(float));

    ncnn::Extractor extractor = impl_->net.create_extractor();
    extractor.set_light_mode(true);
    if (extractor.input("in0", input_mat) != 0)
    {
        set_error(error, "failed to bind input tensor");
        return false;
    }
    ncnn::Mat output_mat;
    if (extractor.extract("out0", output_mat) != 0 || output_mat.empty())
    {
        set_error(error, "failed to extract output tensor");
        return false;
    }

    size_t logical_size = static_cast<size_t>(output_mat.w);
    if (output_mat.dims >= 2) logical_size *= output_mat.h;
    if (output_mat.dims >= 3) logical_size *= output_mat.c;
    if (output_mat.dims >= 4) logical_size *= output_mat.d;
    logical_size *= output_mat.elempack;
    if (output_mat.dims != 2 || output_mat.w != impl_->output_size || output_mat.h != 1 ||
        output_mat.elempack != 1 || logical_size != static_cast<size_t>(impl_->output_size) ||
        output_mat.elemsize / output_mat.elempack != sizeof(float))
    {
        std::ostringstream message;
        message << "unexpected output tensor: dims=" << output_mat.dims
                << " w=" << output_mat.w << " h=" << output_mat.h
                << " c=" << output_mat.c << " total=" << output_mat.total()
                << " elempack=" << output_mat.elempack
                << " elemsize=" << output_mat.elemsize
                << " expected=" << impl_->output_size;
        set_error(error, message.str());
        return false;
    }
    const float* values = reinterpret_cast<const float*>(output_mat.data);
    output->assign(values, values + logical_size);
    return true;
}

} // namespace hunyuan_ocr::diag
