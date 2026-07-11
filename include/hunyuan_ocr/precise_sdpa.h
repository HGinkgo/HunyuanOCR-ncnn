#pragma once

#include <layer.h>

namespace hunyuan_ocr {

ncnn::Layer* create_precise_sdpa_layer(void* userdata);

} // namespace hunyuan_ocr
