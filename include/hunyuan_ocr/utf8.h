#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace hunyuan_ocr {

bool wide_to_utf8(std::wstring_view input, std::string* output, std::string* error);
bool wide_arguments_to_utf8(const std::vector<std::wstring>& input,
                            std::vector<std::string>* output,
                            std::string* error);
std::filesystem::path path_from_utf8(std::string_view value);
std::string path_to_utf8(const std::filesystem::path& path);

} // namespace hunyuan_ocr
