#include "hunyuan_ocr/hunyuan_ocr.h"

#include <filesystem>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

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

    const std::filesystem::path incomplete_model =
        std::filesystem::temp_directory_path() / "hunyuan_ocr_api_incomplete_model";
    std::error_code filesystem_error;
    std::filesystem::remove_all(incomplete_model, filesystem_error);
    filesystem_error.clear();
    std::filesystem::create_directories(incomplete_model, filesystem_error);
    if (!expect(!filesystem_error, "failed to create incomplete model directory"))
    {
        return 4;
    }

    if (!expect(!runtime.load(incomplete_model.string(), runtime_options, &error),
                "incomplete model load must fail") ||
        !expect(error.stage == "model_layout", "incomplete model error stage mismatch") ||
        !expect(!runtime.ready(), "failed load must leave runtime not ready") ||
        !expect(!runtime.layout_report().missing_required.empty(),
                "incomplete model report must list required files"))
    {
        std::filesystem::remove_all(incomplete_model, filesystem_error);
        return 5;
    }
    std::filesystem::remove_all(incomplete_model, filesystem_error);

    hunyuan_ocr::InferenceRequest invalid_request;
    invalid_request.max_tokens = 0;
    if (!expect(!runtime.infer_file("image.png", invalid_request, &result, &error),
                "zero max_tokens must fail") ||
        !expect(error.stage == "request", "max_tokens error stage mismatch"))
    {
        return 6;
    }

    invalid_request = hunyuan_ocr::InferenceRequest();
    invalid_request.prompt_mode = hunyuan_ocr::PromptMode::Custom;
    if (!expect(!runtime.infer_file("image.png", invalid_request, &result, &error),
                "empty custom prompt must fail") ||
        !expect(error.stage == "request", "custom prompt error stage mismatch"))
    {
        return 7;
    }

    invalid_request = hunyuan_ocr::InferenceRequest();
    invalid_request.prompt = "unexpected";
    if (!expect(!runtime.infer_file("image.png", invalid_request, &result, &error),
                "built-in mode with custom text must fail") ||
        !expect(error.stage == "request", "built-in prompt error stage mismatch"))
    {
        return 8;
    }

    if (!expect(!runtime.infer_file("", request, &result, &error),
                "empty image path must fail") ||
        !expect(error.stage == "image_input", "empty image path error stage mismatch"))
    {
        return 9;
    }

    const std::vector<unsigned char> invalid_rgb(11, 0);
    if (!expect(!runtime.infer_rgb(invalid_rgb, 2, 2, request, &result, &error),
                "wrong RGB byte count must fail") ||
        !expect(error.stage == "image_input", "RGB byte count error stage mismatch"))
    {
        return 10;
    }

    const std::vector<unsigned char> empty_rgb;
    if (!expect(!runtime.infer_rgb(empty_rgb,
                                   std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max(),
                                   request,
                                   &result,
                                   &error),
                "overflowing RGB dimensions must fail") ||
        !expect(error.stage == "image_input", "RGB overflow error stage mismatch"))
    {
        return 11;
    }

    const std::vector<unsigned char> valid_rgb(12, 0);
    if (!expect(!runtime.infer_rgb(valid_rgb, 2, 2, request, &result, &error),
                "valid RGB inference before load must fail") ||
        !expect(error.stage == "runtime_state", "valid RGB pre-load error stage mismatch"))
    {
        return 12;
    }

    std::cout << "public runtime API contract passed\n";
    return 0;
}
