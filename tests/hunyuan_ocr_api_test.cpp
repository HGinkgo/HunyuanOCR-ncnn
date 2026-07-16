#include "hunyuan_ocr/hunyuan_ocr.h"

#include <iostream>
#include <type_traits>

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    static_assert(!std::is_copy_constructible<hunyuan_ocr::HunyuanOCR>::value,
                  "runtime must not be copy constructible");
    static_assert(!std::is_copy_assignable<hunyuan_ocr::HunyuanOCR>::value,
                  "runtime must not be copy assignable");
    static_assert(std::is_move_constructible<hunyuan_ocr::HunyuanOCR>::value,
                  "runtime must be move constructible");
    static_assert(std::is_move_assignable<hunyuan_ocr::HunyuanOCR>::value,
                  "runtime must be move assignable");

    const hunyuan_ocr::RuntimeOptions runtime_options;
    if (!expect(runtime_options.num_threads == 0, "default thread count mismatch") ||
        !expect(!runtime_options.vision_vulkan, "vision Vulkan must default off") ||
        !expect(runtime_options.vision_vulkan_device == 0, "default Vulkan device mismatch") ||
        !expect(!runtime_options.dflash, "DFlash must default off") ||
        !expect(runtime_options.repetition_penalty == 1.08f,
                "default repetition penalty mismatch"))
    {
        return 1;
    }

    const hunyuan_ocr::InferenceRequest request;
    if (!expect(request.prompt_mode == hunyuan_ocr::PromptMode::Document,
                "default prompt mode mismatch") ||
        !expect(request.prompt.empty(), "default custom prompt must be empty") ||
        !expect(request.max_tokens == 128, "default max token count mismatch"))
    {
        return 2;
    }

    hunyuan_ocr::HunyuanOCR runtime;
    hunyuan_ocr::InferenceResult result;
    hunyuan_ocr::RuntimeError error;
    if (!expect(!runtime.ready(), "new runtime must not be ready") ||
        !expect(!runtime.infer_file("missing.png", request, &result, &error),
                "inference before load must fail") ||
        !expect(error.stage == "runtime_state", "pre-load error stage mismatch") ||
        !expect(!error.message.empty(), "pre-load error message must not be empty"))
    {
        return 3;
    }

    std::cout << "public runtime API contract passed\n";
    return 0;
}
