#include "hunyuan_ocr/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace hunyuan_ocr {
namespace {

std::string strip_cr(std::string line)
{
    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }
    return line;
}

bool append_utf8(uint32_t codepoint, std::string& out)
{
    if (codepoint < 0x80)
    {
        out.push_back(static_cast<char>(codepoint));
        return true;
    }
    if (codepoint < 0x800)
    {
        out.push_back(static_cast<char>((codepoint >> 6) | 0xC0));
        out.push_back(static_cast<char>((codepoint & 0x3F) | 0x80));
        return true;
    }
    if (codepoint < 0x10000)
    {
        out.push_back(static_cast<char>((codepoint >> 12) | 0xE0));
        out.push_back(static_cast<char>(((codepoint >> 6) & 0x3F) | 0x80));
        out.push_back(static_cast<char>((codepoint & 0x3F) | 0x80));
        return true;
    }
    if (codepoint <= 0x10FFFF)
    {
        out.push_back(static_cast<char>((codepoint >> 18) | 0xF0));
        out.push_back(static_cast<char>(((codepoint >> 12) & 0x3F) | 0x80));
        out.push_back(static_cast<char>(((codepoint >> 6) & 0x3F) | 0x80));
        out.push_back(static_cast<char>((codepoint & 0x3F) | 0x80));
        return true;
    }
    return false;
}

std::string parse_json_string(const std::string& text, size_t& i)
{
    if (i >= text.size() || text[i] != '"')
    {
        throw std::runtime_error("expected JSON string");
    }
    ++i;
    std::string out;
    while (i < text.size())
    {
        const char ch = text[i++];
        if (ch == '"')
        {
            return out;
        }
        if (ch != '\\')
        {
            out.push_back(ch);
            continue;
        }
        if (i >= text.size())
        {
            throw std::runtime_error("truncated JSON escape");
        }
        const char esc = text[i++];
        switch (esc)
        {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
        {
            if (i + 4 > text.size())
            {
                throw std::runtime_error("truncated JSON unicode escape");
            }
            uint32_t codepoint = 0;
            for (int k = 0; k < 4; ++k)
            {
                const char h = text[i++];
                codepoint <<= 4;
                if (h >= '0' && h <= '9') codepoint += static_cast<uint32_t>(h - '0');
                else if (h >= 'a' && h <= 'f') codepoint += static_cast<uint32_t>(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') codepoint += static_cast<uint32_t>(h - 'A' + 10);
                else throw std::runtime_error("invalid JSON unicode escape");
            }
            append_utf8(codepoint, out);
            break;
        }
        default:
            throw std::runtime_error("unsupported JSON escape");
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

std::vector<std::string> parse_json_string_array(const std::string& text)
{
    std::vector<std::string> values;
    size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (i >= text.size() || text[i] != '[')
    {
        throw std::runtime_error("expected JSON array");
    }
    ++i;
    while (i < text.size())
    {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i < text.size() && text[i] == ']')
        {
            ++i;
            return values;
        }
        values.push_back(parse_json_string(text, i));
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i < text.size() && text[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < text.size() && text[i] == ']')
        {
            ++i;
            return values;
        }
        throw std::runtime_error("expected comma or array end");
    }
    throw std::runtime_error("unterminated JSON array");
}

bool is_ascii_digit(char ch)
{
    return ch >= '0' && ch <= '9';
}

bool is_ascii_alpha(char ch)
{
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
}

} // namespace

bool Tokenizer::load(const std::string& vocab_path, const std::string& special_tokens_path, std::string* error)
{
    return load(vocab_path, std::string(), special_tokens_path, error);
}

bool Tokenizer::load(const std::string& vocab_path,
                     const std::string& merges_path,
                     const std::string& special_tokens_path,
                     std::string* error)
{
    id_to_token_.clear();
    token_to_id_.clear();
    merge_ranks_.clear();
    special_token_ids_.clear();
    init_byte_codec();

    std::ifstream vocab(vocab_path);
    if (!vocab.is_open())
    {
        if (error) *error = "failed to open vocab: " + vocab_path;
        return false;
    }

    std::string line;
    while (std::getline(vocab, line))
    {
        std::string token = strip_cr(line);
        const int id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_.emplace(std::move(token), id);
    }
    if (id_to_token_.empty())
    {
        if (error) *error = "vocab is empty: " + vocab_path;
        return false;
    }

    if (!merges_path.empty())
    {
        std::ifstream merges(merges_path);
        if (!merges.is_open())
        {
            if (error) *error = "failed to open merges: " + merges_path;
            return false;
        }

        int rank = 0;
        while (std::getline(merges, line))
        {
            line = strip_cr(line);
            if (line.empty() || line[0] == '#')
            {
                continue;
            }
            const size_t space = line.find(' ');
            if (space == std::string::npos)
            {
                if (error) *error = "invalid merge line: " + line;
                return false;
            }
            const std::string left = line.substr(0, space);
            const std::string right = line.substr(space + 1);
            merge_ranks_.emplace(left + '\x1f' + right, rank++);
        }
    }

    std::ifstream special_file(special_tokens_path);
    if (!special_file.is_open())
    {
        if (error) *error = "failed to open special tokens: " + special_tokens_path;
        return false;
    }

    try
    {
        std::ostringstream buffer;
        buffer << special_file.rdbuf();
        const std::vector<std::string> special_tokens = parse_json_string_array(buffer.str());
        for (const std::string& token : special_tokens)
        {
            const auto it = std::find(id_to_token_.begin(), id_to_token_.end(), token);
            if (it != id_to_token_.end())
            {
                special_token_ids_.insert(static_cast<int>(it - id_to_token_.begin()));
            }
        }
    }
    catch (const std::exception& e)
    {
        if (error) *error = std::string("failed to parse special tokens: ") + e.what();
        return false;
    }

    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text, std::string* error) const
{
    if (!ready())
    {
        if (error) *error = "tokenizer is not loaded";
        return {};
    }
    if (merge_ranks_.empty())
    {
        if (error) *error = "tokenizer merges are not loaded";
        return {};
    }

    std::vector<int> ids;
    size_t begin = 0;
    while (begin < text.size())
    {
        const char ch = text[begin];
        if (is_ascii_digit(ch))
        {
            size_t end = begin;
            while (end < text.size() && is_ascii_digit(text[end]))
            {
                ++end;
            }
            for (size_t chunk = begin; chunk < end; chunk += 3)
            {
                const size_t chunk_size = std::min<size_t>(3, end - chunk);
                std::vector<int> piece_ids = encode_piece(text.substr(chunk, chunk_size), error);
                if (piece_ids.empty() && error && !error->empty())
                {
                    return {};
                }
                ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
            }
            begin = end;
            continue;
        }

        size_t end = begin;
        if (ch == ' ')
        {
            while (end < text.size() && text[end] == ' ')
            {
                ++end;
            }
            if (end < text.size() && is_ascii_alpha(text[end]))
            {
                if (end - begin > 1)
                {
                    std::vector<int> spaces = encode_piece(text.substr(begin, end - begin - 1), error);
                    if (spaces.empty() && error && !error->empty())
                    {
                        return {};
                    }
                    ids.insert(ids.end(), spaces.begin(), spaces.end());
                }

                size_t word_end = end;
                while (word_end < text.size() && is_ascii_alpha(text[word_end]))
                {
                    ++word_end;
                }
                std::vector<int> word_ids = encode_piece(text.substr(end - 1, word_end - end + 1), error);
                if (word_ids.empty() && error && !error->empty())
                {
                    return {};
                }
                ids.insert(ids.end(), word_ids.begin(), word_ids.end());
                begin = word_end;
                continue;
            }
        }
        else
        {
            while (end < text.size() && !is_ascii_digit(text[end]) && text[end] != ' ')
            {
                ++end;
            }
        }

        if (end == begin)
        {
            while (end < text.size() && text[end] == ' ')
            {
                ++end;
            }
        }
        while (end < text.size() && !is_ascii_digit(text[end]) && text[end] != ' ')
        {
            ++end;
        }
        std::vector<int> piece_ids = encode_piece(text.substr(begin, end - begin), error);
        if (piece_ids.empty() && error && !error->empty())
        {
            return {};
        }
        ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
        begin = end;
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const
{
    std::string packed;
    packed.reserve(ids.size() * 3);
    for (const int id : ids)
    {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size()))
        {
            continue;
        }
        if (skip_special_tokens && special_token_ids_.find(id) != special_token_ids_.end())
        {
            continue;
        }

        packed += id_to_token_[id];
    }
    return byte_decode(packed);
}

size_t Tokenizer::vocab_size() const
{
    return id_to_token_.size();
}

bool Tokenizer::ready() const
{
    return !id_to_token_.empty() && !byte_decoder_.empty();
}

void Tokenizer::init_byte_codec()
{
    byte_encoder_.assign(256, std::string());
    byte_decoder_.assign(512, -1);
    auto is_printable = [](int b) {
        return (b >= '!' && b <= '~') || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    };

    int n = 0;
    for (int b = 0; b < 256; ++b)
    {
        const int codepoint = is_printable(b) ? b : 256 + n++;
        if (codepoint >= static_cast<int>(byte_decoder_.size()))
        {
            byte_decoder_.resize(static_cast<size_t>(codepoint + 1), -1);
        }
        std::string encoded;
        append_utf8(static_cast<uint32_t>(codepoint), encoded);
        byte_encoder_[static_cast<size_t>(b)] = std::move(encoded);
        byte_decoder_[static_cast<size_t>(codepoint)] = b;
    }
}

std::vector<int> Tokenizer::encode_piece(const std::string& text, std::string* error) const
{
    std::string byte_token;
    byte_token.reserve(text.size() * 2);
    for (const unsigned char byte : text)
    {
        byte_token += byte_encoder_[static_cast<size_t>(byte)];
    }

    std::vector<int> ids;
    for (const std::string& token : bpe(byte_token))
    {
        const auto it = token_to_id_.find(token);
        if (it == token_to_id_.end())
        {
            if (error) *error = "token not found in vocab: " + token;
            return {};
        }
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) const
{
    std::vector<std::string> word;
    size_t index = 0;
    uint32_t codepoint = 0;
    while (next_utf8(token, index, codepoint))
    {
        std::string symbol;
        append_utf8(codepoint, symbol);
        word.push_back(std::move(symbol));
    }
    if (word.size() <= 1)
    {
        return word;
    }

    while (word.size() > 1)
    {
        int best_rank = std::numeric_limits<int>::max();
        std::string best_left;
        std::string best_right;
        bool found = false;
        for (size_t i = 0; i + 1 < word.size(); ++i)
        {
            const auto it = merge_ranks_.find(word[i] + '\x1f' + word[i + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank)
            {
                best_rank = it->second;
                best_left = word[i];
                best_right = word[i + 1];
                found = true;
            }
        }
        if (!found)
        {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(word.size());
        for (size_t i = 0; i < word.size();)
        {
            if (i + 1 < word.size() && word[i] == best_left && word[i + 1] == best_right)
            {
                merged.push_back(best_left + best_right);
                i += 2;
            }
            else
            {
                merged.push_back(word[i]);
                ++i;
            }
        }
        word = std::move(merged);
    }

    return word;
}

std::string Tokenizer::byte_decode(const std::string& text) const
{
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    uint32_t codepoint = 0;
    while (next_utf8(text, i, codepoint))
    {
        if (codepoint < byte_decoder_.size() && byte_decoder_[static_cast<size_t>(codepoint)] >= 0)
        {
            out.push_back(static_cast<char>(byte_decoder_[static_cast<size_t>(codepoint)]));
        }
    }
    return out;
}

bool Tokenizer::next_utf8(const std::string& text, size_t& index, uint32_t& codepoint)
{
    if (index >= text.size())
    {
        return false;
    }

    const unsigned char c0 = static_cast<unsigned char>(text[index]);
    if (c0 < 0x80)
    {
        codepoint = c0;
        ++index;
        return true;
    }
    if ((c0 >> 5) == 0x6 && index + 1 < text.size())
    {
        const unsigned char c1 = static_cast<unsigned char>(text[index + 1]);
        if ((c1 & 0xC0) != 0x80) return false;
        codepoint = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
        index += 2;
        return true;
    }
    if ((c0 >> 4) == 0xE && index + 2 < text.size())
    {
        const unsigned char c1 = static_cast<unsigned char>(text[index + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[index + 2]);
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return false;
        codepoint = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
        index += 3;
        return true;
    }
    if ((c0 >> 3) == 0x1E && index + 3 < text.size())
    {
        const unsigned char c1 = static_cast<unsigned char>(text[index + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[index + 2]);
        const unsigned char c3 = static_cast<unsigned char>(text[index + 3]);
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
        codepoint = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
        index += 4;
        return true;
    }
    return false;
}

std::vector<int> parse_token_ids(const std::string& text, std::string* error)
{
    std::vector<int> ids;
    std::string number;

    auto flush = [&]() -> bool {
        if (number.empty())
        {
            return true;
        }
        try
        {
            size_t parsed = 0;
            const long value = std::stol(number, &parsed, 10);
            if (parsed != number.size() ||
                value < std::numeric_limits<int>::min() ||
                value > std::numeric_limits<int>::max())
            {
                if (error) *error = "invalid token id: " + number;
                return false;
            }
            ids.push_back(static_cast<int>(value));
        }
        catch (const std::exception&)
        {
            if (error) *error = "invalid token id: " + number;
            return false;
        }
        number.clear();
        return true;
    };

    for (const char ch : text)
    {
        if (std::isdigit(static_cast<unsigned char>(ch)) || (ch == '-' && number.empty()))
        {
            number.push_back(ch);
            continue;
        }
        if (ch == ',' || std::isspace(static_cast<unsigned char>(ch)))
        {
            if (!flush()) return {};
            continue;
        }
        if (error) *error = std::string("unexpected character in token id list: ") + ch;
        return {};
    }
    if (!flush()) return {};
    return ids;
}

} // namespace hunyuan_ocr
