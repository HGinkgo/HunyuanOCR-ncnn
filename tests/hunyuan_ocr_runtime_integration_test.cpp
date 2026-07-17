#include "hunyuan_ocr/hunyuan_ocr.h"

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
        return 4;
    }
    if (first.token_ids.empty() || first.text.empty())
    {
        std::cerr << "first inference returned empty output\n";
        return 5;
    }
    if (first.decoder.mode != hunyuan_ocr::DecoderMode::Autoregressive ||
        first.timing.total_ms <= 0.0)
    {
        std::cerr << "first inference metadata is invalid\n";
        return 6;
    }
    if (callback_after_result_finalized || streamed_token_ids != first.token_ids ||
        streamed_text != first.text)
    {
        std::cerr << "autoregressive stream did not match the committed result\n";
        return 7;
    }

    request.stream_callback = {};
    hunyuan_ocr::InferenceResult second;
    if (!runtime.infer_file(argv[2], request, &second, &error))
    {
        fail("second inference failed", error);
        return 8;
    }
    if (second.token_ids != first.token_ids || second.text != first.text)
    {
        std::cerr << "persistent runtime output changed between requests\n";
        return 9;
    }

    hunyuan_ocr::RuntimeOptions dflash_options = options;
    dflash_options.dflash = true;
    hunyuan_ocr::HunyuanOCR dflash_runtime;
    if (!dflash_runtime.load(argv[1], dflash_options, &error))
    {
        fail("DFlash runtime load failed", error);
        return 10;
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
        return 11;
    }
    if (dflash_result.decoder.mode != hunyuan_ocr::DecoderMode::DFlash ||
        dflash_result.token_ids != first.token_ids || dflash_result.text != first.text)
    {
        std::cerr << "DFlash output did not match autoregressive output\n";
        return 12;
    }
    if (dflash_callback_after_result_finalized ||
        dflash_streamed_token_ids != dflash_result.token_ids ||
        dflash_streamed_text != dflash_result.text)
    {
        std::cerr << "DFlash stream did not match the committed result\n";
        return 13;
    }

    std::cout << "persistent runtime generated " << first.token_ids.size()
              << " exact tokens with autoregressive and DFlash streaming\n";
    return 0;
}
