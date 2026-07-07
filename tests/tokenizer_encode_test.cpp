#include "hunyuan_ocr/tokenizer.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

int hex_value(char ch)
{
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    throw std::runtime_error("invalid hex character");
}

std::string from_hex(const std::string& hex)
{
    if (hex.size() % 2 != 0)
    {
        throw std::runtime_error("odd hex byte string");
    }
    std::string out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2)
    {
        out.push_back(static_cast<char>((hex_value(hex[i]) << 4) | hex_value(hex[i + 1])));
    }
    return out;
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
        std::string text_hex;
        std::vector<int> ids;
    };

    const std::vector<Case> cases = {
        {"ascii", "48656c6c6f2c20776f726c6421", {16883, 11, 2385, 0}},
        {"number_chunk", "31323334", {7827, 19}},
        {"leading_spaces", "20206c656164696e6720616e6420747261696c696e672020", {206, 6255, 289, 40954, 242}},
        {"multi_spaces", "48656c6c6f202020776f726c64", {16883, 242, 2385}},
        {"email_like", "666f6f5f6261722d62617a406578616d706c652e636f6d", {20063, 54361, 2084, 2695, 31402, 3048}},
        {"mixed_newline", "4d6f64656c3a204f4352203132330ae7acace4ba8ce8a18c", {6657, 25, 113184, 206, 7827, 185, 2467, 537}},
        {"zh_en_digits_punctuation", "e4b8ade69687456e676c697368e6b7b7e59088313233efbc8ce6a087e782b9efbc81", {11015, 19405, 9339, 7827, 270, 72859, 2853}},
        {"blank_lines", "6c696e65310a6c696e65320a0a6c696e6534", {2020, 16, 185, 2020, 17, 286, 2020, 19}},
        {"tab_text", "5461627309616e6420737061636573", {98414, 184, 476, 10004}},
        {"symbol_text", "432b2b3137202f206e636e6e3a20667033322d3e746f6b656e733f", {34, 2283, 1374, 1320, 67338, 17678, 25, 51919, 1772, 1336, 101815, 30}},
        {"currency_decimal", "e4bbb7e6a0bce698afefbfa53132332e3435efbc8ce68a98e689a320382e3525", {55131, 100657, 7827, 13, 2445, 270, 29828, 206, 23, 13, 20, 4}},
        {"emoji_text", "656d6f6a6920f09f98802074657374", {20734, 12130, 82457, 208, 1647}},
        {"quote_dash", "e2809c71756f746564e2809d207465787420e280942064617368", {507, 392, 8726, 515, 2602, 3926, 54588}},
        {"literal_backslash_n", "e8a786e9a291e8aeb2e8a7a35c6ee799bde58db7c2b73135e9a298", {9364, 16828, 6087, 1874, 5591, 2383, 952, 871}},
        {"document_prompt_full", "e68f90e58f96e69687e6a1a3e59bbee78987e4b8ade6ada3e69687e79a84e68980e69c89e4bfa1e681afe794a86d61726b646f776ee6a0bce5bc8fe8a1a8e7a4baefbc8ce585b6e4b8ade9a1b5e79c89e38081e9a1b5e8849ae983a8e58886e5bfbde795a5efbc8ce8a1a8e6a0bce794a868746d6ce6a0bce5bc8fe8a1a8e8bebeefbc8ce69687e6a1a3e4b8ade585ace5bc8fe794a86c61746578e6a0bce5bc8fe8a1a8e7a4baefbc8ce68c89e785a7e99885e8afbbe9a1bae5ba8fe7bb84e7bb87e8bf9be8a18ce8a7a3e69e90e38082", {12161, 19177, 12858, 409, 49827, 16317, 1940, 474, 78338, 11971, 2598, 270, 2627, 6319, 19684, 332, 6319, 5705, 2218, 19664, 270, 24784, 474, 13110, 11971, 4088, 270, 19177, 409, 4605, 474, 36078, 11971, 2598, 270, 3802, 6169, 10264, 2297, 1170, 4134, 292}},
        {"spotting_prompt_full", "e6a380e6b58be5b9b6e8af86e588abe59bbee78987e4b8ade79a84e69687e5ad97efbc8ce5b086e69687e69cace59d90e6a087e6a0bce5bc8fe58c96e8be93e587bae38082", {5055, 951, 9977, 12858, 1843, 9738, 270, 934, 8433, 4699, 68216, 8287, 292}},
    };

    for (const Case& test_case : cases)
    {
        error.clear();
        const std::string text = from_hex(test_case.text_hex);
        const std::vector<int> actual = tokenizer.encode(text, &error);
        if (!error.empty())
        {
            std::cerr << test_case.label << " encode failed: " << error << "\n";
            return EXIT_FAILURE;
        }
        if (!expect_equal(actual, test_case.ids, test_case.label))
        {
            return EXIT_FAILURE;
        }
        if (tokenizer.decode(actual, false) != text)
        {
            std::cerr << test_case.label << " decode roundtrip mismatch\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "tokenizer encode cases passed: " << cases.size() << "\n";
    return EXIT_SUCCESS;
}
