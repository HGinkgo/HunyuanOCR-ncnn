#include "hunyuan_ocr/hunyuan_ocr.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

struct ProcessMemory {
    std::uint64_t rss_bytes = 0;
    std::uint64_t rss_anon_bytes = 0;
    std::uint64_t rss_file_bytes = 0;
    std::uint64_t pss_bytes = 0;
    std::uint64_t pss_anon_bytes = 0;
    std::uint64_t pss_file_bytes = 0;
};

void read_kib_field(const std::string& path,
                    const std::string& wanted,
                    std::uint64_t* bytes)
{
#if defined(__linux__)
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream fields(line);
        std::string key;
        std::uint64_t kib = 0;
        std::string unit;
        if (!(fields >> key >> kib >> unit)) continue;
        if (key == wanted)
        {
            *bytes = kib * 1024;
            return;
        }
    }
#else
    (void)path;
    (void)wanted;
    (void)bytes;
#endif
}

ProcessMemory process_memory()
{
    ProcessMemory memory;
    read_kib_field("/proc/self/status", "VmRSS:", &memory.rss_bytes);
    read_kib_field("/proc/self/status", "RssAnon:", &memory.rss_anon_bytes);
    read_kib_field("/proc/self/status", "RssFile:", &memory.rss_file_bytes);
    read_kib_field("/proc/self/smaps_rollup", "Pss:", &memory.pss_bytes);
    read_kib_field("/proc/self/smaps_rollup", "Pss_Anon:", &memory.pss_anon_bytes);
    read_kib_field("/proc/self/smaps_rollup", "Pss_File:", &memory.pss_file_bytes);
    return memory;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: hunyuan_ocr_model_load_probe MODEL_ROOT file|mmap ar|dflash\n";
        return 2;
    }
    const std::string load_mode = argv[2];
    const std::string decoder_mode = argv[3];
    if ((load_mode != "file" && load_mode != "mmap") ||
        (decoder_mode != "ar" && decoder_mode != "dflash"))
    {
        std::cerr << "invalid model load probe mode\n";
        return 2;
    }

    hunyuan_ocr::RuntimeOptions options;
    options.num_threads = 16;
    options.mmap_weights = load_mode == "mmap";
    options.dflash = decoder_mode == "dflash";

    hunyuan_ocr::HunyuanOCR runtime;
    hunyuan_ocr::RuntimeError error;
    const auto start = std::chrono::steady_clock::now();
    if (!runtime.load(argv[1], options, &error))
    {
        std::cerr << "runtime load failed at " << error.stage << ": " << error.message << '\n';
        return 1;
    }
    const auto end = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const ProcessMemory memory = process_memory();

    std::cout << "load_mode=" << load_mode << '\n';
    std::cout << "decoder_mode=" << decoder_mode << '\n';
    std::cout << "load_ms=" << load_ms << '\n';
    std::cout << "mapped_weight_bytes=" << runtime.mapped_weight_bytes() << '\n';
    std::cout << "rss_bytes=" << memory.rss_bytes << '\n';
    std::cout << "rss_anon_bytes=" << memory.rss_anon_bytes << '\n';
    std::cout << "rss_file_bytes=" << memory.rss_file_bytes << '\n';
    std::cout << "pss_bytes=" << memory.pss_bytes << '\n';
    std::cout << "pss_anon_bytes=" << memory.pss_anon_bytes << '\n';
    std::cout << "pss_file_bytes=" << memory.pss_file_bytes << '\n';
    return 0;
}
