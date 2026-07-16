#include "hunyuan_ocr/prompt_builder.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect_equal(const std::vector<int>& actual,
                  const std::vector<int>& expected,
                  const std::string& label)
{
    if (actual == expected)
    {
        return true;
    }

    std::cerr << label << " mismatch\nexpected:";
    for (const int id : expected) std::cerr << ' ' << id;
    std::cerr << "\nactual:";
    for (const int id : actual) std::cerr << ' ' << id;
    std::cerr << "\n";
    return false;
}

} // namespace

int main()
{
    if (std::string(hunyuan_ocr::prompt_mode_name(hunyuan_ocr::PromptMode::Custom)) != "custom")
    {
        std::cerr << "custom prompt mode name mismatch\n";
        return EXIT_FAILURE;
    }
    hunyuan_ocr::PromptBuildResult invalid_custom;
    std::string invalid_custom_error;
    if (hunyuan_ocr::build_hunyuan_ocr_prompt(hunyuan_ocr::PromptMode::Custom,
                                              2,
                                              4,
                                              2,
                                              &invalid_custom,
                                              &invalid_custom_error) ||
        invalid_custom_error.empty())
    {
        std::cerr << "custom mode must use tokenized prompt builder\n";
        return EXIT_FAILURE;
    }

    const std::vector<int> prompt_ids = {16883, 11, 2385, 0}; // "Hello, world!"
    hunyuan_ocr::PromptBuildResult prompt;
    std::string error;
    if (!hunyuan_ocr::build_hunyuan_ocr_prompt_from_tokens(prompt_ids,
                                                           2,
                                                           4,
                                                           2,
                                                           &prompt,
                                                           &error))
    {
        std::cerr << "custom prompt build failed: " << error << "\n";
        return EXIT_FAILURE;
    }

    const std::vector<int> expected_template = {
        120000, 120021, 120118, 120120, 120119, 16883, 11, 2385, 0, 120006,
    };
    const std::vector<int> expected_input_ids = {
        120000, 120021, 120118, 120120, 120120, 120120, 120120, 120120,
        120119, 16883, 11, 2385, 0, 120006,
    };
    if (!expect_equal(prompt.chat_template_ids, expected_template, "chat_template_ids") ||
        !expect_equal(prompt.input_ids, expected_input_ids, "input_ids"))
    {
        return EXIT_FAILURE;
    }

    if (prompt.seq_len != static_cast<int>(expected_input_ids.size()) ||
        prompt.vision_token_count != 5 ||
        prompt.image_token_id != 120120)
    {
        std::cerr << "prompt metadata mismatch\n";
        return EXIT_FAILURE;
    }

    const int seq_len = prompt.seq_len;
    const std::vector<int> expected_axis1 = {0, 1, 2, 3, 0, 1, 2, 7, 8, 9, 10, 11, 12, 13};
    const std::vector<int> expected_axis2 = {0, 1, 2, 3, 0, 0, 0, 7, 8, 9, 10, 11, 12, 13};
    std::vector<int> axis1(prompt.position_ids.begin() + seq_len,
                           prompt.position_ids.begin() + 2 * seq_len);
    std::vector<int> axis2(prompt.position_ids.begin() + 2 * seq_len,
                           prompt.position_ids.begin() + 3 * seq_len);
    if (!expect_equal(axis1, expected_axis1, "position_axis1") ||
        !expect_equal(axis2, expected_axis2, "position_axis2"))
    {
        return EXIT_FAILURE;
    }

    std::cout << "prompt builder custom prompt passed\n";
    return EXIT_SUCCESS;
}
