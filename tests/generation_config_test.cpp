#include "hunyuan_ocr/generation_config.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool write_text(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream file(path);
    file << text;
    return file.good();
}

} // namespace

int main()
{
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "hunyuan_ocr_eos_ids_test.json";
    std::vector<int> eos_ids;
    std::string error;

    if (!write_text(path, "{\"eos_ids\":[120020]}\n") ||
        !hunyuan_ocr::load_eos_ids(path.string(), &eos_ids, &error))
    {
        return 1;
    }
    if (eos_ids != std::vector<int>{120020} ||
        hunyuan_ocr::is_eos_token(120007, eos_ids) ||
        !hunyuan_ocr::is_eos_token(120020, eos_ids))
    {
        return 2;
    }

    if (!write_text(path, "{\n  \"eos_ids\": [120007, 120020]\n}\n") ||
        !hunyuan_ocr::load_eos_ids(path.string(), &eos_ids, &error))
    {
        return 3;
    }
    if (!hunyuan_ocr::is_eos_token(120007, eos_ids) ||
        !hunyuan_ocr::is_eos_token(120020, eos_ids))
    {
        return 4;
    }

    if (!write_text(path, "{\"eos_ids\": []}\n") ||
        hunyuan_ocr::load_eos_ids(path.string(), &eos_ids, &error))
    {
        return 5;
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}
