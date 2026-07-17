#include "mapped_model_file.h"

#include "hunyuan_ocr/utf8.h"

#include <net.h>

#include <cstring>
#include <limits>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hunyuan_ocr {

class MappedModelFile::Impl {
public:
    ~Impl()
    {
#if defined(_WIN32)
        if (data != nullptr)
        {
            UnmapViewOfFile(data);
        }
        if (mapping != nullptr)
        {
            CloseHandle(mapping);
        }
        if (file != INVALID_HANDLE_VALUE)
        {
            CloseHandle(file);
        }
#else
        if (data != nullptr)
        {
            munmap(const_cast<unsigned char*>(data), size);
        }
        if (file >= 0)
        {
            ::close(file);
        }
#endif
    }

    const unsigned char* data = nullptr;
    size_t size = 0;
#if defined(_WIN32)
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int file = -1;
#endif
};

MappedModelFile::MappedModelFile()
    : impl_(new Impl)
{
}

MappedModelFile::~MappedModelFile() = default;
MappedModelFile::MappedModelFile(MappedModelFile&&) noexcept = default;
MappedModelFile& MappedModelFile::operator=(MappedModelFile&&) noexcept = default;

bool MappedModelFile::open(const std::filesystem::path& path, std::string* error)
{
    if (error)
    {
        error->clear();
    }

    std::unique_ptr<Impl> mapped(new Impl);
#if defined(_WIN32)
    mapped->file = CreateFileW(path.c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ,
                               nullptr,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
    if (mapped->file == INVALID_HANDLE_VALUE)
    {
        if (error)
        {
            *error = "failed to open model file: " + path_to_utf8(path) +
                " (Windows error " + std::to_string(GetLastError()) + ")";
        }
        return false;
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(mapped->file, &file_size) || file_size.QuadPart <= 0 ||
        static_cast<unsigned long long>(file_size.QuadPart) >
            static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
    {
        if (error)
        {
            *error = "model file is empty or too large to map: " + path_to_utf8(path);
        }
        return false;
    }
    mapped->size = static_cast<size_t>(file_size.QuadPart);
    mapped->mapping = CreateFileMappingW(mapped->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapped->mapping == nullptr)
    {
        if (error)
        {
            *error = "failed to create model file mapping: " + path_to_utf8(path) +
                " (Windows error " + std::to_string(GetLastError()) + ")";
        }
        return false;
    }
    mapped->data = static_cast<const unsigned char*>(
        MapViewOfFile(mapped->mapping, FILE_MAP_READ, 0, 0, 0));
    if (mapped->data == nullptr)
    {
        if (error)
        {
            *error = "failed to map model file: " + path_to_utf8(path) +
                " (Windows error " + std::to_string(GetLastError()) + ")";
        }
        return false;
    }
#else
    mapped->file = ::open(path.c_str(), O_RDONLY);
    if (mapped->file < 0)
    {
        if (error)
        {
            *error = "failed to open model file: " + path_to_utf8(path) +
                " (" + std::strerror(errno) + ")";
        }
        return false;
    }

    struct stat file_stat{};
    if (fstat(mapped->file, &file_stat) != 0 || file_stat.st_size <= 0 ||
        static_cast<unsigned long long>(file_stat.st_size) >
            static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
    {
        if (error)
        {
            *error = "model file is empty or too large to map: " + path_to_utf8(path);
        }
        return false;
    }
    mapped->size = static_cast<size_t>(file_stat.st_size);
    void* view = mmap(nullptr, mapped->size, PROT_READ, MAP_PRIVATE, mapped->file, 0);
    if (view == MAP_FAILED)
    {
        if (error)
        {
            *error = "failed to map model file: " + path_to_utf8(path) +
                " (" + std::strerror(errno) + ")";
        }
        return false;
    }
    mapped->data = static_cast<const unsigned char*>(view);
#endif

    impl_ = std::move(mapped);
    return true;
}

void MappedModelFile::close()
{
    impl_.reset(new Impl);
}

bool MappedModelFile::ready() const
{
    return impl_ != nullptr && impl_->data != nullptr && impl_->size > 0;
}

const unsigned char* MappedModelFile::data() const
{
    return ready() ? impl_->data : nullptr;
}

size_t MappedModelFile::size() const
{
    return ready() ? impl_->size : 0;
}

MappedModelDataReader::MappedModelDataReader(const unsigned char* data, size_t size)
    : data_(data), size_(size)
{
}

size_t MappedModelDataReader::read(void* buffer, size_t size) const
{
    if (buffer == nullptr || data_ == nullptr || offset_ > size_ || size > size_ - offset_)
    {
        return 0;
    }
    std::memcpy(buffer, data_ + offset_, size);
    offset_ += size;
    return size;
}

size_t MappedModelDataReader::reference(size_t size, const void** buffer) const
{
    if (buffer == nullptr)
    {
        return 0;
    }
    *buffer = nullptr;
    if (data_ == nullptr || offset_ > size_ || size > size_ - offset_)
    {
        return 0;
    }
    *buffer = data_ + offset_;
    offset_ += size;
    return size;
}

size_t MappedModelDataReader::offset() const
{
    return offset_;
}

bool load_model_file(ncnn::Net& net,
                     const std::filesystem::path& path,
                     bool use_mmap,
                     std::shared_ptr<MappedModelFile>* mapped_file,
                     std::string* error)
{
    if (!use_mmap)
    {
        if (mapped_file)
        {
            mapped_file->reset();
        }
        if (net.load_model(path.c_str()) != 0)
        {
            if (error) *error = "failed to load model file: " + path_to_utf8(path);
            return false;
        }
        return true;
    }
    if (mapped_file == nullptr)
    {
        if (error) *error = "mapped model owner pointer is null";
        return false;
    }

    std::shared_ptr<MappedModelFile> mapped(new MappedModelFile);
    if (!mapped->open(path, error))
    {
        return false;
    }
    MappedModelDataReader reader(mapped->data(), mapped->size());
    if (net.load_model(reader) != 0)
    {
        if (error) *error = "failed to load mapped model file: " + path_to_utf8(path);
        return false;
    }
    if (reader.offset() != mapped->size())
    {
        if (error)
        {
            *error = "mapped model file was not fully consumed: " + path_to_utf8(path) +
                " (consumed " + std::to_string(reader.offset()) + " of " +
                std::to_string(mapped->size()) + " bytes)";
        }
        return false;
    }
    *mapped_file = std::move(mapped);
    return true;
}

} // namespace hunyuan_ocr
