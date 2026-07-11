#include "hunyuan_ocr/text_runtime.h"

#include <cmath>
#include <vector>

int main()
{
    const std::vector<int> position_ids = {978, 811, 621, 431};
    const std::vector<float> actual = hunyuan_ocr::detail::build_mrope(position_ids, 1, true);
    if (actual.size() != 128)
    {
        return 1;
    }

    const int oracle_dims[] = {0, 2, 31, 32, 63, 64, 95, 96, 127};
    const float oracle_cos[] = {
        -0.569676042f,
        -0.0682837293f,
        0.929675043f,
        0.970673621f,
        1.0f,
        0.510228872f,
        0.97144419f,
        0.991688073f,
        1.0f,
    };
    for (int index = 0; index < 9; ++index)
    {
        if (std::fabs(actual[oracle_dims[index]] - oracle_cos[index]) > 2e-7f)
        {
            return 2;
        }
    }
    return 0;
}
