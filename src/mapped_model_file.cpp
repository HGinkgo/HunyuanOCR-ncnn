#include "mapped_model_file.h"

#include "hunyuan_ocr/utf8.h"

#include <net.h>

#include <cstring>
#include <fstream>
#include <limits>
#include <new>
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

bool SharedModelData::open(const std::filesystem::path& path,
                           bool use_mmap,
                           std::string* error)
{
    close();
    if (error)
    {
        error->clear();
    }

    if (use_mmap)
    {
        std::shared_ptr<MappedModelFile> mapped(new MappedModelFile);
        if (!mapped->open(path, error))
        {
            return false;
        }
        mapped_file_ = std::move(mapped);
        return true;
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        if (error) *error = "failed to open model file: " + path_to_utf8(path);
        return false;
    }

    const std::streamoff end = input.tellg();
    if (end <= 0 ||
        static_cast<unsigned long long>(end) >
            static_cast<unsigned long long>(std::numeric_limits<size_t>::max()) ||
        static_cast<unsigned long long>(end) >
            static_cast<unsigned long long>(std::numeric_limits<std::streamsize>::max()))
    {
        if (error) *error = "model file is empty or too large to load: " + path_to_utf8(path);
        return false;
    }

    owned_size_ = static_cast<size_t>(end);
    owned_data_.reset(new (std::nothrow) unsigned char[owned_size_]);
    if (!owned_data_)
    {
        owned_size_ = 0;
        if (error) *error = "failed to allocate model data: " + path_to_utf8(path);
        return false;
    }
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(owned_data_.get()),
               static_cast<std::streamsize>(owned_size_));
    if (!input)
    {
        owned_data_.reset();
        owned_size_ = 0;
        if (error) *error = "failed to read model file: " + path_to_utf8(path);
        return false;
    }
    return true;
}

void SharedModelData::close()
{
    mapped_file_.reset();
    owned_data_.reset();
    owned_size_ = 0;
}

bool SharedModelData::ready() const
{
    return (mapped_file_ && mapped_file_->ready()) || owned_data_ != nullptr;
}

const unsigned char* SharedModelData::data() const
{
    if (mapped_file_ && mapped_file_->ready())
    {
        return mapped_file_->data();
    }
    return owned_data_.get();
}

size_t SharedModelData::size() const
{
    if (mapped_file_ && mapped_file_->ready())
    {
        return mapped_file_->size();
    }
    return owned_size_;
}

size_t SharedModelData::mapped_bytes() const
{
    return mapped_file_ && mapped_file_->ready() ? mapped_file_->size() : 0;
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

bool load_shared_model_data(ncnn::Net& net,
                            const SharedModelData& model_data,
                            std::string* error)
{
    if (!model_data.ready())
    {
        if (error) *error = "shared model data is not ready";
        return false;
    }

    MappedModelDataReader reader(model_data.data(), model_data.size());
    if (net.load_model(reader) != 0)
    {
        if (error) *error = "failed to load shared model data";
        return false;
    }
    if (reader.offset() != model_data.size())
    {
        if (error)
        {
            *error = "shared model data was not fully consumed (consumed " +
                std::to_string(reader.offset()) + " of " +
                std::to_string(model_data.size()) + " bytes)";
        }
        return false;
    }
    return true;
}

} // namespace hunyuan_ocr
