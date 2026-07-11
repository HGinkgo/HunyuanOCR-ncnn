#include "hunyuan_ocr/utf8.h"

#include <cstdint>
#include <utility>

namespace hunyuan_ocr {
namespace {

void append_utf8(uint32_t codepoint, std::string* output)
{
    if (codepoint <= 0x7f)
    {
        output->push_back(static_cast<char>(codepoint));
    }
    else if (codepoint <= 0x7ff)
    {
        output->push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    else if (codepoint <= 0xffff)
    {
        output->push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    else
    {
        output->push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

bool fail_conversion(const char* message, std::string* output, std::string* error)
{
    if (output) output->clear();
    if (error) *error = message;
    return false;
}

} // namespace

bool wide_to_utf8(std::wstring_view input, std::string* output, std::string* error)
{
    if (output == nullptr)
    {
        if (error) *error = "UTF-8 output pointer is null";
        return false;
    }

    output->clear();
    if (error) error->clear();
    output->reserve(input.size() * 3);
    for (size_t i = 0; i < input.size(); ++i)
    {
        uint32_t codepoint = static_cast<uint32_t>(input[i]);
        if (codepoint >= 0xd800 && codepoint <= 0xdbff)
        {
            if (i + 1 >= input.size())
            {
                return fail_conversion("unpaired high UTF-16 surrogate", output, error);
            }
            const uint32_t low = static_cast<uint32_t>(input[++i]);
            if (low < 0xdc00 || low > 0xdfff)
            {
                return fail_conversion("invalid UTF-16 surrogate pair", output, error);
            }
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
        }
        else if (codepoint >= 0xdc00 && codepoint <= 0xdfff)
        {
            return fail_conversion("unpaired low UTF-16 surrogate", output, error);
        }
        else if (codepoint > 0x10ffff)
        {
            return fail_conversion("wide character is outside the Unicode range", output, error);
        }
        append_utf8(codepoint, output);
    }
    return true;
}

bool wide_arguments_to_utf8(const std::vector<std::wstring>& input,
                            std::vector<std::string>* output,
                            std::string* error)
{
    if (output == nullptr)
    {
        if (error) *error = "UTF-8 argument output pointer is null";
        return false;
    }
    output->clear();
    output->reserve(input.size());
    for (const std::wstring& argument : input)
    {
        std::string converted;
        if (!wide_to_utf8(argument, &converted, error))
        {
            output->clear();
            return false;
        }
        output->push_back(std::move(converted));
    }
    return true;
}

std::filesystem::path path_from_utf8(std::string_view value)
{
    return std::filesystem::u8path(value.begin(), value.end());
}

std::string path_to_utf8(const std::filesystem::path& path)
{
    return path.u8string();
}

} // namespace hunyuan_ocr
