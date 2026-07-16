#include "hunyuan_ocr/hunyuan_ocr.h"

#include <option.h>
#include <platform.h>

namespace hunyuan_ocr {

namespace {

void clear_error(RuntimeError* error)
{
    if (error)
    {
        *error = RuntimeError();
    }
}

bool fail(RuntimeError* error, const char* stage, const std::string& message)
{
    if (error)
    {
        error->stage = stage;
        error->message = message;
    }
    return false;
}

} // namespace

class HunyuanOCR::Impl {
public:
    bool ready = false;
    RuntimeOptions options;
    ModelLayoutReport layout_report;
};

std::string project_version()
{
#ifdef HUNYUAN_OCR_VERSION
    return HUNYUAN_OCR_VERSION;
#else
    return "unknown";
#endif
}

std::string ncnn_version()
{
#ifdef NCNN_VERSION_STRING
    return NCNN_VERSION_STRING;
#else
    return "unknown";
#endif
}

ncnn::Option make_fp32_ncnn_option(int num_threads)
{
    ncnn::Option option;
    if (num_threads > 0)
    {
        option.num_threads = num_threads;
    }
    option.use_vulkan_compute = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    // Runtime Nets are long-lived, but their inference buffers are request-scoped.
    option.use_local_pool_allocator = false;
    return option;
}

HunyuanOCR::HunyuanOCR()
    : impl_(new Impl)
{
}

HunyuanOCR::~HunyuanOCR() = default;
HunyuanOCR::HunyuanOCR(HunyuanOCR&&) noexcept = default;
HunyuanOCR& HunyuanOCR::operator=(HunyuanOCR&&) noexcept = default;

bool HunyuanOCR::load(const std::string& model_root)
{
    return load(model_root, RuntimeOptions(), nullptr);
}

bool HunyuanOCR::load(const std::string& model_root,
                      const RuntimeOptions& options,
                      RuntimeError* error)
{
    clear_error(error);
    if (!impl_)
    {
        impl_.reset(new Impl);
    }

    impl_->ready = false;
    impl_->options = options;
    impl_->layout_report = check_model_layout(model_root);
    if (!impl_->layout_report.required_files_present())
    {
        return fail(error, "model_layout", "required model files are missing");
    }

    impl_->ready = true;
    return true;
}

bool HunyuanOCR::infer_file(const std::string&,
                            const InferenceRequest&,
                            InferenceResult* result,
                            RuntimeError* error)
{
    if (result)
    {
        *result = InferenceResult();
    }
    clear_error(error);
    if (!impl_ || !impl_->ready)
    {
        return fail(error, "runtime_state", "runtime is not loaded");
    }
    return fail(error, "runtime_state", "runtime inference is not initialized");
}

bool HunyuanOCR::infer_rgb(const std::vector<unsigned char>&,
                           int,
                           int,
                           const InferenceRequest&,
                           InferenceResult* result,
                           RuntimeError* error)
{
    if (result)
    {
        *result = InferenceResult();
    }
    clear_error(error);
    if (!impl_ || !impl_->ready)
    {
        return fail(error, "runtime_state", "runtime is not loaded");
    }
    return fail(error, "runtime_state", "runtime inference is not initialized");
}

bool HunyuanOCR::ready() const
{
    return impl_ && impl_->ready;
}

const ModelLayoutReport& HunyuanOCR::layout_report() const
{
    static const ModelLayoutReport empty_report;
    return impl_ ? impl_->layout_report : empty_report;
}

} // namespace hunyuan_ocr
