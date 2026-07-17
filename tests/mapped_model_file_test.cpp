#include "mapped_model_file.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <utility>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    static_assert(!std::is_copy_constructible<hunyuan_ocr::MappedModelFile>::value,
                  "mapped files must not be copy constructible");
    static_assert(std::is_move_constructible<hunyuan_ocr::MappedModelFile>::value,
                  "mapped files must be move constructible");

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "hunyuan_ocr_mapped_model_file_test";
    std::error_code filesystem_error;
    std::filesystem::remove_all(root, filesystem_error);
    filesystem_error.clear();
    std::filesystem::create_directories(root, filesystem_error);
    if (!expect(!filesystem_error, "failed to create mapped-file test directory"))
    {
        return 1;
    }

    std::string error;
    hunyuan_ocr::MappedModelFile mapped;
    if (!expect(!mapped.open(root / "missing.bin", &error),
                "missing mapped file must fail") ||
        !expect(!error.empty(), "missing mapped file must report an error") ||
        !expect(!mapped.ready(), "failed mapping must not be ready"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 2;
    }

    const std::filesystem::path empty_path = root / "empty.bin";
    std::ofstream(empty_path, std::ios::binary);
    error.clear();
    if (!expect(!mapped.open(empty_path, &error), "empty mapped file must fail") ||
        !expect(!error.empty(), "empty mapped file must report an error"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 3;
    }

    const std::array<unsigned char, 8> expected{{0x10, 0x20, 0x30, 0x40,
                                                  0x50, 0x60, 0x70, 0x80}};
    const std::filesystem::path model_path = root / "weights.bin";
    {
        std::ofstream output(model_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(expected.data()),
                     static_cast<std::streamsize>(expected.size()));
    }

    error.clear();
    if (!expect(mapped.open(model_path, &error), "valid mapped file must open") ||
        !expect(error.empty(), "valid mapped file must not report an error") ||
        !expect(mapped.ready(), "valid mapped file must be ready") ||
        !expect(mapped.size() == expected.size(), "mapped file size mismatch") ||
        !expect(mapped.data()[0] == expected[0] && mapped.data()[7] == expected[7],
                "mapped file content mismatch"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 4;
    }

    hunyuan_ocr::MappedModelFile moved = std::move(mapped);
    if (!expect(!mapped.ready(), "moved-from mapping must not be ready") ||
        !expect(moved.ready(), "moved mapping must remain ready"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 5;
    }

    hunyuan_ocr::MappedModelDataReader reader(moved.data(), moved.size());
    std::array<unsigned char, 3> copied{{0, 0, 0}};
    if (!expect(reader.read(copied.data(), copied.size()) == copied.size(),
                "bounded reader copy failed") ||
        !expect(copied[0] == expected[0] && copied[2] == expected[2],
                "bounded reader copy mismatch") ||
        !expect(reader.offset() == copied.size(), "bounded reader copy offset mismatch"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 6;
    }

    const void* referenced = nullptr;
    if (!expect(reader.reference(2, &referenced) == 2, "bounded reader reference failed") ||
        !expect(referenced == moved.data() + 3, "bounded reader reference pointer mismatch") ||
        !expect(reader.offset() == 5, "bounded reader reference offset mismatch"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 7;
    }

    referenced = moved.data();
    if (!expect(reader.reference(4, &referenced) == 0,
                "out-of-bounds reference must fail") ||
        !expect(referenced == nullptr, "failed reference must clear its pointer") ||
        !expect(reader.offset() == 5, "failed reference must not advance offset") ||
        !expect(reader.read(copied.data(), 4) == 0, "out-of-bounds read must fail") ||
        !expect(reader.offset() == 5, "failed read must not advance offset"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 8;
    }

    moved.close();
    if (!expect(!moved.ready(), "closed mapping must not be ready") ||
        !expect(moved.data() == nullptr && moved.size() == 0,
                "closed mapping must clear its view"))
    {
        std::filesystem::remove_all(root, filesystem_error);
        return 9;
    }

    std::filesystem::remove_all(root, filesystem_error);
    std::cout << "mapped model file lifecycle passed\n";
    return 0;
}
