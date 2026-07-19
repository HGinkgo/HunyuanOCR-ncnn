#include "hunyuan_ocr/hunyuan_ocr.h"

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model-dir> <image>\n";
        return 1;
    }

    hunyuan_ocr::RuntimeError error;
    hunyuan_ocr::HunyuanOCR runtime;
    if (!runtime.load(argv[1], {}, &error))
    {
        std::cerr << "Failed to load OCR runtime at " << error.stage
                  << ": " << error.message << "\n";
        return 2;
    }

    hunyuan_ocr::InferenceRequest request;
    request.prompt_mode = hunyuan_ocr::PromptMode::Document;
    request.max_tokens = 8192;
    request.stream_callback = [](const hunyuan_ocr::InferenceChunk& chunk) {
        if (!chunk.text_delta.empty())
        {
            std::cout << chunk.text_delta << std::flush;
        }
    };

    hunyuan_ocr::InferenceResult result;
    if (!runtime.infer_file(argv[2], request, &result, &error))
    {
        std::cerr << "Failed to run OCR at " << error.stage
                  << ": " << error.message << "\n";
        return 3;
    }

    std::cout << '\n';
    return 0;
}
