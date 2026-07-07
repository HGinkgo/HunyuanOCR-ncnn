#include "hunyuan_ocr/tokenizer.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string path_join(const std::string& lhs, const std::string& rhs)
{
    if (lhs.empty() || lhs.back() == '/')
    {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

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

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: hunyuan_ocr_tokenizer_test <tokenizer_dir>\n";
        return EXIT_FAILURE;
    }

    const std::string tokenizer_dir = argv[1];
    std::string error;
    hunyuan_ocr::Tokenizer tokenizer;
    if (!tokenizer.load(path_join(tokenizer_dir, "vocab.txt"),
                        path_join(tokenizer_dir, "merges.txt"),
                        path_join(tokenizer_dir, "special_tokens.json"),
                        &error))
    {
        std::cerr << "load tokenizer failed: " << error << "\n";
        return EXIT_FAILURE;
    }

    struct Case {
        std::string label;
        std::string text;
        std::vector<int> ids;
    };

    const std::vector<Case> cases = {
        {"ascii", "Hello, world!", {16883, 11, 2385, 0}},
        {"number_chunk", "1234", {7827, 19}},
        {"leading_spaces", "  leading and trailing  ", {206, 6255, 289, 40954, 242}},
        {"ocr_prompt",
         "\xE6\xA3\x80\xE6\xB5\x8B\xE5\xB9\xB6\xE8\xAF\x86\xE5\x88\xAB"
         "\xE5\x9B\xBE\xE7\x89\x87\xE4\xB8\xAD\xE7\x9A\x84\xE6\x96\x87\xE5\xAD\x97",
         {5055, 951, 9977, 12858, 1843, 9738}},
        {"document_prompt",
         "\xE6\x8F\x90\xE5\x8F\x96\xE6\x96\x87\xE6\xA1\xA3\xE5\x9B\xBE\xE7\x89\x87"
         "\xE4\xB8\xAD\xE6\xAD\xA3\xE6\x96\x87\xE7\x9A\x84\xE6\x89\x80\xE6\x9C\x89"
         "\xE4\xBF\xA1\xE6\x81\xAF\xE7\x94\xA8markdown\xE6\xA0\xBC\xE5\xBC\x8F"
         "\xE8\xA1\xA8\xE7\xA4\xBA",
         {12161, 19177, 12858, 409, 49827, 16317, 1940, 474, 78338, 11971, 2598}},
        {"mixed_newline",
         "Model: OCR 123\n\xE7\xAC\xAC\xE4\xBA\x8C\xE8\xA1\x8C",
         {6657, 25, 113184, 206, 7827, 185, 2467, 537}},
    };

    for (const Case& test_case : cases)
    {
        error.clear();
        const std::vector<int> actual = tokenizer.encode(test_case.text, &error);
        if (!error.empty())
        {
            std::cerr << test_case.label << " encode failed: " << error << "\n";
            return EXIT_FAILURE;
        }
        if (!expect_equal(actual, test_case.ids, test_case.label))
        {
            return EXIT_FAILURE;
        }
        if (tokenizer.decode(actual, false) != test_case.text)
        {
            std::cerr << test_case.label << " decode roundtrip mismatch\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "tokenizer encode cases passed: " << cases.size() << "\n";
    return EXIT_SUCCESS;
}
