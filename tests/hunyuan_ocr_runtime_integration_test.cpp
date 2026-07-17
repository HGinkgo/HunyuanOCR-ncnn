#include "hunyuan_ocr/hunyuan_ocr.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

bool fail(const std::string& message, const hunyuan_ocr::RuntimeError& error)
{
    std::cerr << message;
    if (!error.stage.empty() || !error.message.empty())
    {
        std::cerr << ": " << error.stage << ": " << error.message;
    }
    std::cerr << '\n';
    return false;
}

std::uintmax_t expected_mapped_weight_bytes(const std::filesystem::path& model_root)
{
    return std::filesystem::file_size(model_root / "vision" / "vision.ncnn.bin") +
           std::filesystem::file_size(model_root / "text_embed" / "text_embed.ncnn.bin") +
           std::filesystem::file_size(
               model_root / "text_decoder" / "text_decoder_kv.ncnn.bin") +
           std::filesystem::file_size(model_root / "lm_head" / "lm_head.ncnn.bin") +
           std::filesystem::file_size(model_root / "dflash" / "dflash.ncnn.bin");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: hunyuan_ocr_runtime_integration_test MODEL_ROOT IMAGE\n";
        return 2;
    }

    hunyuan_ocr::RuntimeOptions options;
    options.num_threads = 16;

    hunyuan_ocr::RuntimeError error;
    hunyuan_ocr::HunyuanOCR runtime;
    if (!runtime.load(argv[1], options, &error))
    {
        fail("runtime load failed", error);
        return 3;
    }
    if (runtime.mapped_weight_bytes() != 0)
    {
        std::cerr << "default runtime unexpectedly mapped model weights\n";
        return 4;
    }

    hunyuan_ocr::InferenceRequest request;
    request.prompt_mode = hunyuan_ocr::PromptMode::Spotting;
    request.max_tokens = 8;

    hunyuan_ocr::InferenceResult first;
    std::vector<int> streamed_token_ids;
    std::string streamed_text;
    bool callback_after_result_finalized = false;
    request.stream_callback = [&](const hunyuan_ocr::InferenceChunk& chunk) {
        callback_after_result_finalized =
            callback_after_result_finalized || !first.token_ids.empty() || !first.text.empty();
        streamed_token_ids.push_back(chunk.token_id);
        streamed_text += chunk.text_delta;
    };
    if (!runtime.infer_file(argv[2], request, &first, &error))
    {
        fail("first inference failed", error);
        return 5;
    }
    if (first.token_ids.empty() || first.text.empty())
    {
        std::cerr << "first inference returned empty output\n";
        return 6;
    }
    if (first.decoder.mode != hunyuan_ocr::DecoderMode::Autoregressive ||
        first.timing.total_ms <= 0.0)
    {
        std::cerr << "first inference metadata is invalid\n";
        return 7;
    }
    if (callback_after_result_finalized || streamed_token_ids != first.token_ids ||
        streamed_text != first.text)
    {
        std::cerr << "autoregressive stream did not match the committed result\n";
        return 8;
    }

    request.stream_callback = {};
    hunyuan_ocr::InferenceResult second;
    if (!runtime.infer_file(argv[2], request, &second, &error))
    {
        fail("second inference failed", error);
        return 9;
    }
    if (second.token_ids != first.token_ids || second.text != first.text)
    {
        std::cerr << "persistent runtime output changed between requests\n";
        return 10;
    }

    hunyuan_ocr::RuntimeOptions dflash_options = options;
    dflash_options.dflash = true;
    dflash_options.mmap_weights = true;
    hunyuan_ocr::HunyuanOCR dflash_runtime;
    if (!dflash_runtime.load(argv[1], dflash_options, &error))
    {
        fail("DFlash runtime load failed", error);
        return 11;
    }
    const std::uintmax_t expected_mapped_bytes = expected_mapped_weight_bytes(argv[1]);
    if (dflash_runtime.mapped_weight_bytes() != expected_mapped_bytes)
    {
        std::cerr << "mmap runtime retained " << dflash_runtime.mapped_weight_bytes()
                  << " mapped bytes, expected " << expected_mapped_bytes << '\n';
        return 12;
    }

    hunyuan_ocr::InferenceResult dflash_result;
    std::vector<int> dflash_streamed_token_ids;
    std::string dflash_streamed_text;
    bool dflash_callback_after_result_finalized = false;
    request.stream_callback = [&](const hunyuan_ocr::InferenceChunk& chunk) {
        dflash_callback_after_result_finalized = dflash_callback_after_result_finalized ||
            !dflash_result.token_ids.empty() || !dflash_result.text.empty();
        dflash_streamed_token_ids.push_back(chunk.token_id);
        dflash_streamed_text += chunk.text_delta;
    };
    if (!dflash_runtime.infer_file(argv[2], request, &dflash_result, &error))
    {
        fail("DFlash inference failed", error);
        return 13;
    }
    if (dflash_result.decoder.mode != hunyuan_ocr::DecoderMode::DFlash ||
        dflash_result.token_ids != first.token_ids || dflash_result.text != first.text)
    {
        std::cerr << "DFlash output did not match autoregressive output\n";
        return 14;
    }
    if (dflash_callback_after_result_finalized ||
        dflash_streamed_token_ids != dflash_result.token_ids ||
        dflash_streamed_text != dflash_result.text)
    {
        std::cerr << "DFlash stream did not match the committed result\n";
        return 15;
    }

    std::cout << "persistent runtime generated " << first.token_ids.size()
              << " exact tokens with autoregressive and DFlash streaming\n";
    return 0;
}
