#include "decoder_microbench.h"

#include "hunyuan_ocr/hunyuan_ocr.h"
#include "precise_sdpa_profile.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <net.h>

namespace {

void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program
        << " --model PATH [--output-csv PATH] [--env-out PATH]\n"
        << "             [--warmup N] [--repeat N]\n"
        << "             [--context-lens LIST] [--query-lens LIST] [--threads LIST]\n"
        << "\n"
        << "Target decoder microbenchmark for AR query=1 vs DFlash verify query=16.\n"
        << "Measures production run_decoder_step only. Not registered as CTest.\n";
}

bool write_text_file(const std::string& path, const std::string& text, std::string* error)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
    {
        if (error) *error = "failed to open output: " + path;
        return false;
    }
    file << text;
    if (!file.good())
    {
        if (error) *error = "failed to write output: " + path;
        return false;
    }
    return true;
}

std::string collect_env_text(const std::string& model_root, int warmup, int repeat)
{
    std::ostringstream stream;
    stream << "project_version=" << hunyuan_ocr::project_version() << '\n';
    stream << "ncnn_version=" << hunyuan_ocr::ncnn_version() << '\n';
    stream << "model_root=" << model_root << '\n';
    stream << "benchmark=target_decoder_microbench\n";
    stream << "warmup=" << warmup << '\n';
    stream << "repeat=" << repeat << '\n';
    stream << "measured=detail::run_decoder_step only "
              "(extractor create + input bind + forward + hidden/KV extract)\n";
    stream << "excluded=model_load,input_prep,validate_outputs,output_destructors,"
              "text_embed,lm_head,token_select,draft,commit\n";
    stream << "net_policy=one decoder net per thread count; destroy before next threads\n";
#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
    stream << "sdpa_profile=ON\n";
#else
    stream << "sdpa_profile=OFF\n";
#endif
    return stream.str();
}

} // namespace

int main(int argc, char** argv)
{
    std::string model_root;
    std::string output_csv = "/tmp/dflash_verify_microbenchmark_v2/results.csv";
    std::string env_out = "/tmp/dflash_verify_microbenchmark_v2/env.txt";
    std::string sdpa_csv = "/tmp/dflash_sdpa_profile/sdpa_profile.csv";
    int warmup = 3;
    int repeat = 7;
    std::string context_text = "256,512,1024";
    std::string query_text = "1,16";
    std::string threads_text = "4,8,16,32";
    std::string cache_mode_text = "exact";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc)
            {
                std::cerr << name << " requires a value\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--model")
        {
            model_root = need_value("--model");
            continue;
        }
        if (arg == "--output-csv")
        {
            output_csv = need_value("--output-csv");
            continue;
        }
        if (arg == "--env-out")
        {
            env_out = need_value("--env-out");
            continue;
        }
        if (arg == "--sdpa-csv")
        {
            sdpa_csv = need_value("--sdpa-csv");
            continue;
        }
        if (arg == "--warmup")
        {
            warmup = std::stoi(need_value("--warmup"));
            continue;
        }
        if (arg == "--repeat")
        {
            repeat = std::stoi(need_value("--repeat"));
            continue;
        }
        if (arg == "--context-lens")
        {
            context_text = need_value("--context-lens");
            continue;
        }
        if (arg == "--query-lens")
        {
            query_text = need_value("--query-lens");
            continue;
        }
        if (arg == "--threads")
        {
            threads_text = need_value("--threads");
            continue;
        }
        if (arg == "--cache-mode")
        {
            cache_mode_text = need_value("--cache-mode");
            continue;
        }
        std::cerr << "unknown argument: " << arg << '\n';
        return 1;
    }

    hunyuan_ocr::detail::DecoderMicrobenchCacheMode cache_mode =
        hunyuan_ocr::detail::DecoderMicrobenchCacheMode::Exact;
    if (cache_mode_text == "exact")
    {
        cache_mode = hunyuan_ocr::detail::DecoderMicrobenchCacheMode::Exact;
    }
    else if (cache_mode_text == "reserved")
    {
        cache_mode = hunyuan_ocr::detail::DecoderMicrobenchCacheMode::Reserved;
    }
    else
    {
        std::cerr << "--cache-mode must be exact or reserved\n";
        return 1;
    }

    if (model_root.empty())
    {
        std::cerr << "--model is required\n";
        return 1;
    }
    if (warmup < 0 || repeat <= 0)
    {
        std::cerr << "warmup must be non-negative and repeat must be positive\n";
        return 1;
    }

    std::string error;
    std::vector<int> context_lens;
    std::vector<int> query_lens;
    std::vector<int> thread_counts;
    if (!hunyuan_ocr::detail::parse_positive_int_list(context_text, &context_lens, &error) ||
        !hunyuan_ocr::detail::parse_positive_int_list(query_text, &query_lens, &error) ||
        !hunyuan_ocr::detail::parse_positive_int_list(threads_text, &thread_counts, &error))
    {
        std::cerr << error << '\n';
        return 1;
    }

    // Group by thread count so only one decoder net is live at a time.
    std::vector<hunyuan_ocr::detail::DecoderMicrobenchRow> rows;
    std::vector<std::string> sdpa_rows;
    rows.reserve(context_lens.size() * query_lens.size() * thread_counts.size());
    for (int num_threads : thread_counts)
    {
        std::unique_ptr<ncnn::Net> decoder_net(new ncnn::Net);
        if (!hunyuan_ocr::detail::load_text_decoder_kv_net(
                model_root, num_threads, decoder_net.get(), &error))
        {
            std::cerr << "failed to load decoder for threads=" << num_threads
                      << ": " << error << '\n';
            return 1;
        }

        for (int context_len : context_lens)
        {
            for (int query_len : query_lens)
            {
                hunyuan_ocr::detail::DecoderMicrobenchCell cell;
                cell.context_len = context_len;
                cell.query_len = query_len;
                cell.num_threads = num_threads;
                cell.warmup = warmup;
                cell.repeat = repeat;
                cell.cache_mode = cache_mode;

                hunyuan_ocr::detail::DecoderMicrobenchRow row;
                std::string sdpa_row;
                if (!hunyuan_ocr::detail::run_decoder_microbench_cell(
                        *decoder_net, cell, &row, &error, &sdpa_row))
                {
                    std::cerr << "microbench cell failed"
                              << " context_len=" << cell.context_len
                              << " query_len=" << cell.query_len
                              << " num_threads=" << cell.num_threads
                              << " cache_mode="
                              << hunyuan_ocr::detail::decoder_microbench_cache_mode_name(
                                     cell.cache_mode)
                              << ": " << error << '\n';
                    return 1;
                }
                rows.push_back(row);
                if (!sdpa_row.empty())
                {
                    sdpa_rows.push_back(sdpa_row);
                }
                std::cout << "context_len=" << row.cell.context_len
                          << " query_len=" << row.cell.query_len
                          << " num_threads=" << row.cell.num_threads
                          << " cache_mode="
                          << hunyuan_ocr::detail::decoder_microbench_cache_mode_name(
                                 row.cell.cache_mode)
                          << " median_ms=" << row.timing.median_ms
                          << " per_query_token_ms=" << row.timing.per_query_token_ms
                          << '\n';
            }
        }
        // Destroy this thread's decoder net before loading the next one.
        decoder_net.reset();
    }

    hunyuan_ocr::detail::sort_decoder_microbench_rows(&rows);

    std::ostringstream csv;
    csv << hunyuan_ocr::detail::decoder_microbench_csv_header() << '\n';
    for (const auto& row : rows)
    {
        csv << hunyuan_ocr::detail::format_decoder_microbench_csv_row(row) << '\n';
    }

    if (!write_text_file(env_out, collect_env_text(model_root, warmup, repeat), &error) ||
        !write_text_file(output_csv, csv.str(), &error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "wrote " << rows.size() << " rows to " << output_csv << '\n';

#if defined(HUNYUAN_OCR_ENABLE_SDPA_PROFILE)
    if (sdpa_rows.size() != rows.size())
    {
        std::cerr << "SDPA profile row count mismatch\n";
        return 1;
    }
    // Keep SDPA CSV aligned with sorted decoder CSV order.
    // Sort profile rows by their generated context/query/thread keys.
    std::sort(sdpa_rows.begin(), sdpa_rows.end(), [](const std::string& a, const std::string& b) {
        auto parse_key = [](const std::string& line) {
            int ctx = 0, q = 0, th = 0;
            char comma = 0;
            std::istringstream in(line);
            in >> ctx >> comma >> q >> comma >> th;
            return std::tuple<int, int, int>(ctx, q, th);
        };
        return parse_key(a) < parse_key(b);
    });
    std::ostringstream sdpa_out;
    sdpa_out << hunyuan_ocr::detail::precise_sdpa_profile_csv_header() << '\n';
    for (const auto& line : sdpa_rows)
    {
        sdpa_out << line << '\n';
    }
    if (!write_text_file(sdpa_csv, sdpa_out.str(), &error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "wrote " << sdpa_rows.size() << " SDPA profile rows to " << sdpa_csv << '\n';
#else
    (void)sdpa_csv;
    (void)sdpa_rows;
#endif
    return 0;
}
