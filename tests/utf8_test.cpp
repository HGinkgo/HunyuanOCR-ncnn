#include "hunyuan_ocr/utf8.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << "\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    std::string error;
    std::string utf8;
    if (!expect(hunyuan_ocr::wide_to_utf8(L"\u4e2d\u6587 Prompt", &utf8, &error),
                "Chinese wide string conversion failed") ||
        !expect(utf8 == "\xe4\xb8\xad\xe6\x96\x87 Prompt", "Chinese UTF-8 bytes mismatch"))
    {
        return 1;
    }

    const std::wstring surrogate_pair = {
        static_cast<wchar_t>(0xd83d),
        static_cast<wchar_t>(0xde00),
    };
    if (!expect(hunyuan_ocr::wide_to_utf8(surrogate_pair, &utf8, &error),
                "UTF-16 surrogate pair conversion failed") ||
        !expect(utf8 == "\xf0\x9f\x98\x80", "surrogate pair UTF-8 bytes mismatch"))
    {
        return 1;
    }

    const std::wstring invalid_surrogate = {static_cast<wchar_t>(0xd83d)};
    error.clear();
    if (!expect(!hunyuan_ocr::wide_to_utf8(invalid_surrogate, &utf8, &error),
                "unpaired UTF-16 surrogate must fail") ||
        !expect(!error.empty(), "invalid UTF-16 must report an error"))
    {
        return 1;
    }

    std::vector<std::string> arguments;
    const std::vector<std::wstring> wide_arguments = {
        L"hunyuan_ocr_cli.exe",
        L"--prompt",
        L"\u53ea\u8f93\u51fa\u53ef\u89c1\u6587\u5b57",
    };
    if (!expect(hunyuan_ocr::wide_arguments_to_utf8(wide_arguments, &arguments, &error),
                "wide argument conversion failed") ||
        !expect(arguments.size() == 3, "wide argument count mismatch") ||
        !expect(arguments[2] == "\xe5\x8f\xaa\xe8\xbe\x93\xe5\x87\xba\xe5\x8f\xaf\xe8\xa7\x81\xe6\x96\x87\xe5\xad\x97",
                "Chinese prompt argument mismatch"))
    {
        return 1;
    }

    const std::string filename = "hunyuan_utf8_\xe4\xb8\xad\xe6\x96\x87_\xf0\x9f\x98\x80.tmp";
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / hunyuan_ocr::path_from_utf8(filename);
    if (!expect(hunyuan_ocr::path_to_utf8(path.filename()) == filename, "UTF-8 path round trip failed"))
    {
        return 1;
    }
    {
        std::ofstream output(path, std::ios::binary);
        if (!expect(output.is_open(), "failed to create a Unicode path"))
        {
            return 1;
        }
        output << "ok";
    }
    if (!expect(std::filesystem::is_regular_file(path), "Unicode path file was not created"))
    {
        return 1;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (!expect(!ec, "failed to remove Unicode path test file"))
    {
        return 1;
    }

    return 0;
}
