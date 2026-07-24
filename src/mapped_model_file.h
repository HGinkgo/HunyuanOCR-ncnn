#pragma once

#include <datareader.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace ncnn {
class Net;
}

namespace hunyuan_ocr {

class MappedModelFile {
public:
    MappedModelFile();
    ~MappedModelFile();
    MappedModelFile(MappedModelFile&&) noexcept;
    MappedModelFile& operator=(MappedModelFile&&) noexcept;
    MappedModelFile(const MappedModelFile&) = delete;
    MappedModelFile& operator=(const MappedModelFile&) = delete;

    bool open(const std::filesystem::path& path, std::string* error);
    void close();
    bool ready() const;
    const unsigned char* data() const;
    size_t size() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class MappedModelDataReader : public ncnn::DataReader {
public:
    MappedModelDataReader(const unsigned char* data, size_t size);

    size_t read(void* buffer, size_t size) const override;
    size_t reference(size_t size, const void** buffer) const override;
    size_t offset() const;

private:
    const unsigned char* data_;
    size_t size_;
    mutable size_t offset_ = 0;
};

class SharedModelData {
public:
    SharedModelData() = default;
    SharedModelData(const SharedModelData&) = delete;
    SharedModelData& operator=(const SharedModelData&) = delete;

    bool open(const std::filesystem::path& path, bool use_mmap, std::string* error);
    void close();
    bool ready() const;
    const unsigned char* data() const;
    size_t size() const;
    size_t mapped_bytes() const;

private:
    std::shared_ptr<MappedModelFile> mapped_file_;
    std::unique_ptr<unsigned char[]> owned_data_;
    size_t owned_size_ = 0;
};

bool load_model_file(ncnn::Net& net,
                     const std::filesystem::path& path,
                     bool use_mmap,
                     std::shared_ptr<MappedModelFile>* mapped_file,
                     std::string* error);

bool load_shared_model_data(ncnn::Net& net,
                            const SharedModelData& model_data,
                            std::string* error);

} // namespace hunyuan_ocr
