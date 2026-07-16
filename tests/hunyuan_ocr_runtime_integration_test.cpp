#include "hunyuan_ocr/hunyuan_ocr.h"

#include <iostream>

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

    hunyuan_ocr::InferenceResult second;
    if (!runtime.infer_file(argv[2], request, &second, &error))
    {
        fail("second inference failed", error);
        return 7;
    }
    if (second.token_ids != first.token_ids || second.text != first.text)
    {
        std::cerr << "persistent runtime output changed between requests\n";
        return 8;
    }

    std::cout << "persistent runtime generated " << first.token_ids.size()
              << " exact tokens twice\n";
    return 0;
}
