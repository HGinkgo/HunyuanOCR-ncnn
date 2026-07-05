#include "hunyuan_ocr/hunyuan_ocr.h"

#include <option.h>
#include <platform.h>

namespace hunyuan_ocr {

std::string ncnn_version()
{
#ifdef NCNN_VERSION_STRING
    return NCNN_VERSION_STRING;
#else
    return "unknown";
#endif
}

ncnn::Option make_fp32_ncnn_option()
{
    ncnn::Option option;
    option.use_vulkan_compute = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
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

