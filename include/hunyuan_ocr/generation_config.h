#pragma once

#include <string>
#include <vector>

namespace hunyuan_ocr {

bool load_eos_ids(const std::string& path, std::vector<int>* eos_ids, std::string* error);
bool is_eos_token(int token_id, const std::vector<int>& eos_ids);

} // namespace hunyuan_ocr
