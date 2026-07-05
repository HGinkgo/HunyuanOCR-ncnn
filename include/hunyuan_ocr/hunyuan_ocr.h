#pragma once

#include "hunyuan_ocr/model_layout.h"

#include <string>

namespace ncnn {
class Option;
}

namespace hunyuan_ocr {

std::string ncnn_version();
ncnn::Option make_fp32_ncnn_option();

class HunyuanOCR {
public:
    bool load(const std::string& model_root);
    bool ready() const;
    const ModelLayoutReport& layout_report() const;

private:
    bool ready_ = false;
    ModelLayoutReport layout_report_;
};

} // namespace hunyuan_ocr

