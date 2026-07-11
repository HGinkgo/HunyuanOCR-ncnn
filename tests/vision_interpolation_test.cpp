#include "hunyuan_ocr/vision_runtime.h"

#include <cmath>

int main()
{
    const float actual = hunyuan_ocr::detail::bilinear_source_coordinate(17, 128, 54);
    const float expected = (17.5f * 128.0f / 54.0f) - 0.5f;
    return std::fabs(actual - expected) <= 1e-6f ? 0 : 1;
}
