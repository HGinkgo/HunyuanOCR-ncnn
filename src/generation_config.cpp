#include "hunyuan_ocr/generation_config.h"
#include "hunyuan_ocr/utf8.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iterator>

namespace hunyuan_ocr {
namespace {

bool fail(const std::string& message, std::string* error)
{
    if (error) *error = message;
    return false;
}

void skip_space(const std::string& text, size_t* index)
{
    while (*index < text.size() && std::isspace(static_cast<unsigned char>(text[*index])))
    {
        ++*index;
    }
}

} // namespace

bool load_eos_ids(const std::string& path, std::vector<int>* eos_ids, std::string* error)
{
    if (eos_ids == nullptr)
    {
        return fail("eos_ids output is null", error);
    }

    std::ifstream file(path_from_utf8(path));
    if (!file.is_open())
    {
        return fail("failed to open EOS config: " + path, error);
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const size_t key = text.find("\"eos_ids\"");
    if (key == std::string::npos)
    {
        return fail("EOS config is missing eos_ids: " + path, error);
    }
    size_t index = text.find('[', key);
    if (index == std::string::npos)
    {
        return fail("EOS config eos_ids is not an array: " + path, error);
    }
    ++index;

    std::vector<int> parsed;
    while (true)
    {
        skip_space(text, &index);
        if (index >= text.size())
        {
            return fail("unterminated eos_ids array: " + path, error);
        }
        if (text[index] == ']')
        {
            ++index;
            break;
        }

        int value = 0;
        const char* begin = text.data() + index;
        const char* end = text.data() + text.size();
        const std::from_chars_result result = std::from_chars(begin, end, value);
        if (result.ec != std::errc() || result.ptr == begin || value < 0)
        {
            return fail("invalid EOS token id in: " + path, error);
        }
        parsed.push_back(value);
        index = static_cast<size_t>(result.ptr - text.data());

        skip_space(text, &index);
        if (index >= text.size())
        {
            return fail("unterminated eos_ids array: " + path, error);
        }
        if (text[index] == ',')
        {
            ++index;
            continue;
        }
        if (text[index] != ']')
        {
            return fail("expected ',' or ']' in eos_ids: " + path, error);
        }
        ++index;
        break;
    }

    if (parsed.empty())
    {
        return fail("EOS config eos_ids must not be empty: " + path, error);
    }
    std::sort(parsed.begin(), parsed.end());
    parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());
    *eos_ids = std::move(parsed);
    return true;
}

bool is_eos_token(int token_id, const std::vector<int>& eos_ids)
{
    return std::binary_search(eos_ids.begin(), eos_ids.end(), token_id);
}

} // namespace hunyuan_ocr
