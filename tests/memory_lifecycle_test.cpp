#include "hunyuan_ocr/hunyuan_ocr.h"

#include <mat.h>
#include <net.h>
#include <option.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr int kInferenceLoops = 100;
constexpr std::uint64_t kMaxRssGrowthBytes = 64ull * 1024ull * 1024ull;

constexpr bool address_sanitizer_enabled()
{
#if defined(__SANITIZE_ADDRESS__)
    return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}

std::uint64_t resident_set_bytes()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    {
        return 0;
    }
    return static_cast<std::uint64_t>(counters.WorkingSetSize);
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    {
        return 0;
    }
    return static_cast<std::uint64_t>(info.resident_size);
#else
    std::ifstream statm("/proc/self/statm");
    std::uint64_t total_pages = 0;
    std::uint64_t resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages))
    {
        return 0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        return 0;
    }
    return resident_pages * static_cast<std::uint64_t>(page_size);
#endif
}

class TinyRuntime {
public:
    explicit TinyRuntime(const ncnn::Option& option)
    {
        net_.opt = option;
    }

    bool load()
    {
        static const char param[] =
            "7767517\n"
            "2 2\n"
            "Input input 0 1 data\n"
            "Pooling pooling 1 1 data output 0=0 1=1 2=1\n";
        return net_.load_param_mem(param) == 0 &&
               net_.load_model(reinterpret_cast<const unsigned char*>("")) == 0;
    }

    bool infer(int width, int height, int channels)
    {
        ncnn::Mat input(width, height, channels);
        if (input.empty())
        {
            return false;
        }
        input.fill(1.0f);

        ncnn::Extractor extractor = net_.create_extractor();
        if (extractor.input("data", input) != 0)
        {
            return false;
        }
        ncnn::Mat output;
        if (extractor.extract("output", output) != 0)
        {
            return false;
        }
        return output.w == width && output.h == height && output.c == channels;
    }

private:
    ncnn::Net net_;
};

int fail(const char* message, int code)
{
    std::cerr << message << '\n';
    return code;
}

} // namespace

int main()
{
    const ncnn::Option option = hunyuan_ocr::make_fp32_ncnn_option(1);
    TinyRuntime image_runtime(option);
    TinyRuntime text_runtime(option);
    if (!image_runtime.load() || !text_runtime.load())
    {
        return fail("failed to load lifecycle test networks", 1);
    }

    if (!image_runtime.infer(512, 256, 3) || !text_runtime.infer(1024, 64, 1))
    {
        return fail("lifecycle warmup inference failed", 2);
    }
    const std::uint64_t rss_before = resident_set_bytes();
    if (rss_before == 0)
    {
        return fail("failed to read process RSS", 3);
    }

    for (int i = 0; i < kInferenceLoops; ++i)
    {
        if (!image_runtime.infer(514 + i * 2, 256, 3) ||
            !text_runtime.infer(1028 + i * 4, 64, 1))
        {
            return fail("100-loop lifecycle inference failed", 4);
        }
    }

    const std::uint64_t rss_after = resident_set_bytes();
    if (rss_after == 0)
    {
        return fail("failed to read process RSS after inference", 5);
    }
    const std::uint64_t rss_growth = rss_after > rss_before ? rss_after - rss_before : 0;
    std::cout << "inference_loops=" << kInferenceLoops << '\n'
              << "rss_before_bytes=" << rss_before << '\n'
              << "rss_after_bytes=" << rss_after << '\n'
              << "rss_growth_bytes=" << rss_growth << '\n'
              << "rss_limit_enforced=" << !address_sanitizer_enabled() << '\n';

    if (option.use_local_pool_allocator)
    {
        return fail("runtime option retains request buffers in the Net-local allocator", 6);
    }
    // ASAN deliberately quarantines freed blocks, so its process RSS does not
    // measure whether the runtime returned request-scoped allocations.
    if (!address_sanitizer_enabled() && rss_growth > kMaxRssGrowthBytes)
    {
        return fail("RSS grew beyond the 64 MiB lifecycle bound", 7);
    }
    return 0;
}
