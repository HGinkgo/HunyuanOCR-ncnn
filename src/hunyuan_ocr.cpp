#include "hunyuan_ocr/hunyuan_ocr.h"

#include <option.h>
#include <platform.h>

namespace hunyuan_ocr {

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

bool HunyuanOCR::load(const std::string& model_root)
{
    layout_report_ = check_model_layout(model_root);
    ready_ = layout_report_.required_files_present();
    return ready_;
}

bool HunyuanOCR::ready() const
{
    return ready_;
}

const ModelLayoutReport& HunyuanOCR::layout_report() const
{
    return layout_report_;
}

} // namespace hunyuan_ocr
