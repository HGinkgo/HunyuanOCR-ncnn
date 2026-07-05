#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace hunyuan_ocr {

class Tokenizer {
public:
    bool load(const std::string& vocab_path, const std::string& special_tokens_path, std::string* error);
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;
    size_t vocab_size() const;
    bool ready() const;

private:
    void init_byte_decoder();
    std::string byte_decode(const std::string& text) const;
    static bool next_utf8(const std::string& text, size_t& index, uint32_t& codepoint);

    std::vector<std::string> id_to_token_;
    std::unordered_set<int> special_token_ids_;
    std::vector<int> byte_decoder_;
};

std::vector<int> parse_token_ids(const std::string& text, std::string* error);

} // namespace hunyuan_ocr

