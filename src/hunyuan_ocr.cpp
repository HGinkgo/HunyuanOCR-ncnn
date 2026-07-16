#include "hunyuan_ocr/hunyuan_ocr.h"

#include "hunyuan_ocr/image_preprocessor.h"
#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/tokenizer.h"
#include "hunyuan_ocr/utf8.h"
#include "hunyuan_ocr/vision_runtime.h"

#include <option.h>
#include <platform.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <utility>

namespace hunyuan_ocr {

namespace {

void clear_error(RuntimeError* error)
{
    if (error)
    {
        *error = RuntimeError();
    }
}

bool fail(RuntimeError* error, const char* stage, const std::string& message)
{
    if (error)
    {
        error->stage = stage;
        error->message = message;
    }
    return false;
}

bool validate_request(const InferenceRequest& request, RuntimeError* error)
{
    if (request.max_tokens <= 0)
    {
        return fail(error, "request", "max_tokens must be positive");
    }
    if (request.prompt_mode == PromptMode::Custom)
    {
        if (request.prompt.empty())
        {
            return fail(error, "request", "custom prompt must not be empty");
        }
    }
    else if (!request.prompt.empty())
    {
        return fail(error, "request", "custom prompt text requires custom prompt mode");
    }
    return true;
}

bool regular_file_exists(const std::filesystem::path& path)
{
    std::error_code filesystem_error;
    return std::filesystem::is_regular_file(path, filesystem_error);
}

bool resolve_dynamic_vision_paths(const std::string& model_root,
                                  std::string* param_path,
                                  std::string* bin_path,
                                  std::string* pos_embed_path)
{
    const std::filesystem::path vision_dir = path_from_utf8(model_root) / "vision";
    const std::filesystem::path pos = vision_dir / "pos_embed.bin";
    const std::filesystem::path param = vision_dir / "vision.ncnn.param";
    const std::filesystem::path bin = vision_dir / "vision.ncnn.bin";
    if (regular_file_exists(param) && regular_file_exists(bin) && regular_file_exists(pos))
    {
        *param_path = path_to_utf8(param);
        *bin_path = path_to_utf8(bin);
        *pos_embed_path = path_to_utf8(pos);
        return true;
    }

    const std::filesystem::path legacy_param = vision_dir / "vision_dynamic.ncnn.param";
    const std::filesystem::path legacy_bin = vision_dir / "vision_dynamic.ncnn.bin";
    if (!regular_file_exists(legacy_param) || !regular_file_exists(legacy_bin) ||
        !regular_file_exists(pos))
    {
        return false;
    }
    *param_path = path_to_utf8(legacy_param);
    *bin_path = path_to_utf8(legacy_bin);
    *pos_embed_path = path_to_utf8(pos);
    return true;
}

bool resolve_fixed_vision_paths(const std::string& model_root,
                                int grid_h,
                                int grid_w,
                                std::string* param_path,
                                std::string* bin_path)
{
    const std::string grid_name =
        "grid_" + std::to_string(grid_h) + "x" + std::to_string(grid_w);
    const std::filesystem::path vision_dir =
        path_from_utf8(model_root) / "vision" / grid_name;
    const std::filesystem::path param = vision_dir / "vision.ncnn.param";
    const std::filesystem::path bin = vision_dir / "vision.ncnn.bin";
    if (!regular_file_exists(param) || !regular_file_exists(bin))
    {
        return false;
    }
    *param_path = path_to_utf8(param);
    *bin_path = path_to_utf8(bin);
    return true;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

class HunyuanOCR::Impl {
public:
    using VisionKey = std::pair<int, int>;

    std::unique_ptr<VisionRuntime> make_vision_runtime() const
    {
        VisionRuntimeOptions vision_options;
        vision_options.num_threads = options.num_threads;
        vision_options.use_vulkan = options.vision_vulkan;
        vision_options.vulkan_device = options.vision_vulkan_device;
        return std::unique_ptr<VisionRuntime>(new VisionRuntime(vision_options));
    }

    VisionRuntime* fixed_vision_runtime(int grid_h, int grid_w, RuntimeError* error)
    {
        const VisionKey key(grid_h, grid_w);
        const auto existing = fixed_vision_runtimes.find(key);
        if (existing != fixed_vision_runtimes.end())
        {
            return existing->second.get();
        }

        std::string param_path;
        std::string bin_path;
        if (!resolve_fixed_vision_paths(model_root,
                                        grid_h,
                                        grid_w,
                                        &param_path,
                                        &bin_path))
        {
            fail(error,
                 "vision_load",
                 "no dynamic or fixed-grid vision model is available for grid_" +
                     std::to_string(grid_h) + "x" + std::to_string(grid_w));
            return nullptr;
        }

        std::unique_ptr<VisionRuntime> runtime = make_vision_runtime();
        std::string runtime_error;
        if (!runtime->load(param_path, bin_path, &runtime_error))
        {
            fail(error, "vision_load", runtime_error);
            return nullptr;
        }
        VisionRuntime* loaded = runtime.get();
        fixed_vision_runtimes.emplace(key, std::move(runtime));
        return loaded;
    }

    bool infer_preprocessed(const ImagePreprocessResult& image,
                            const InferenceRequest& request,
                            InferenceResult* result,
                            RuntimeError* error)
    {
        const auto vision_start = std::chrono::steady_clock::now();
        VisionRuntime* active_vision = dynamic_vision.get();
        if (active_vision == nullptr)
        {
            active_vision = fixed_vision_runtime(image.grid_h, image.grid_w, error);
            if (active_vision == nullptr)
            {
                return false;
            }
        }

        const int merge_size = preprocessor.config().merge_size;
        const int patch_h = image.grid_h / merge_size;
        const int patch_w = image.grid_w / merge_size;
        const int vision_token_count = patch_h * (patch_w + 1) + 2;
        VisionRuntimeResult vision;
        std::string runtime_error;
        const bool vision_ok = dynamic_vision
            ? active_vision->run_dynamic_pixel_values(image.pixel_values,
                                                      image.grid_h,
                                                      image.grid_w,
                                                      merge_size,
                                                      &vision,
                                                      &runtime_error)
            : active_vision->run_pixel_values(image.pixel_values,
                                              image.patch_count,
                                              vision_token_count,
                                              &vision,
                                              &runtime_error);
        if (!vision_ok)
        {
            return fail(error, "vision", runtime_error);
        }
        result->timing.vision_ms = elapsed_ms(vision_start, std::chrono::steady_clock::now());

        PromptBuildResult prompt;
        if (request.prompt_mode == PromptMode::Custom)
        {
            runtime_error.clear();
            const std::vector<int> prompt_tokens = tokenizer.encode(request.prompt, &runtime_error);
            if (!runtime_error.empty())
            {
                return fail(error, "tokenizer", runtime_error);
            }
            if (!build_hunyuan_ocr_prompt_from_tokens(prompt_tokens,
                                                      image.grid_h,
                                                      image.grid_w,
                                                      merge_size,
                                                      &prompt,
                                                      &runtime_error))
            {
                return fail(error, "prompt", runtime_error);
            }
        }
        else if (!build_hunyuan_ocr_prompt(request.prompt_mode,
                                           image.grid_h,
                                           image.grid_w,
                                           merge_size,
                                           &prompt,
                                           &runtime_error))
        {
            return fail(error, "prompt", runtime_error);
        }

        const auto text_start = std::chrono::steady_clock::now();
        const std::vector<int> no_expected_tokens;
        if (options.dflash)
        {
            DFlashDecodeResult decode;
            if (!text_runtime->run_vlm_dflash_decode_with_prompt(
                    prompt.input_ids,
                    prompt.position_ids,
                    prompt.image_token_id,
                    vision.vision_features,
                    vision.vision_token_count,
                    no_expected_tokens,
                    request.max_tokens,
                    options.repetition_penalty,
                    &decode,
                    &runtime_error))
            {
                return fail(error, "text_generation", runtime_error);
            }
            result->token_ids = std::move(decode.decode.generated_tokens);
            result->decoder.mode = DecoderMode::DFlash;
            result->decoder.block_count = decode.block_count;
            result->decoder.drafted_token_count = decode.drafted_token_count;
            result->decoder.accepted_draft_token_count = decode.accepted_draft_token_count;
        }
        else
        {
            TextDecodeResult decode;
            if (!text_runtime->run_vlm_decode_with_prompt(prompt.input_ids,
                                                          prompt.position_ids,
                                                          prompt.image_token_id,
                                                          vision.vision_features,
                                                          vision.vision_token_count,
                                                          no_expected_tokens,
                                                          request.max_tokens,
                                                          options.repetition_penalty,
                                                          &decode,
                                                          &runtime_error))
            {
                return fail(error, "text_generation", runtime_error);
            }
            result->token_ids = std::move(decode.generated_tokens);
            result->decoder.mode = DecoderMode::Autoregressive;
        }
        result->timing.text_ms = elapsed_ms(text_start, std::chrono::steady_clock::now());
        result->text = tokenizer.decode(result->token_ids, true);
        return true;
    }

    bool ready = false;
    RuntimeOptions options;
    ModelLayoutReport layout_report;
    std::string model_root;
    ImagePreprocessor preprocessor;
    Tokenizer tokenizer;
    std::unique_ptr<VisionRuntime> dynamic_vision;
    std::map<VisionKey, std::unique_ptr<VisionRuntime>> fixed_vision_runtimes;
    std::unique_ptr<TextRuntime> text_runtime;
};

std::string project_version()
{
#ifdef HUNYUAN_OCR_VERSION
    return HUNYUAN_OCR_VERSION;
#else
    return "unknown";
#endif
}

std::string ncnn_version()
{
#ifdef NCNN_VERSION_STRING
    return NCNN_VERSION_STRING;
#else
    return "unknown";
#endif
}

ncnn::Option make_fp32_ncnn_option(int num_threads)
{
    ncnn::Option option;
    if (num_threads > 0)
    {
        option.num_threads = num_threads;
    }
    option.use_vulkan_compute = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;
    // Runtime Nets are long-lived, but their inference buffers are request-scoped.
    option.use_local_pool_allocator = false;
    return option;
}

HunyuanOCR::HunyuanOCR()
    : impl_(new Impl)
{
}

HunyuanOCR::~HunyuanOCR() = default;
HunyuanOCR::HunyuanOCR(HunyuanOCR&&) noexcept = default;
HunyuanOCR& HunyuanOCR::operator=(HunyuanOCR&&) noexcept = default;

bool HunyuanOCR::load(const std::string& model_root)
{
    return load(model_root, RuntimeOptions(), nullptr);
}

bool HunyuanOCR::load(const std::string& model_root,
                      const RuntimeOptions& options,
                      RuntimeError* error)
{
    clear_error(error);
    impl_.reset(new Impl);
    impl_->options = options;
    impl_->model_root = model_root;
    impl_->layout_report = check_model_layout(model_root);

    if (options.num_threads < 0)
    {
        return fail(error, "runtime_options", "num_threads must not be negative");
    }
    if (options.vision_vulkan_device < 0)
    {
        return fail(error, "runtime_options", "vision Vulkan device must not be negative");
    }
    if (!std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0.0f)
    {
        return fail(error, "runtime_options", "repetition penalty must be positive and finite");
    }
    if (!impl_->layout_report.required_files_present())
    {
        return fail(error, "model_layout", "required model files are missing");
    }

    const std::filesystem::path root = path_from_utf8(model_root);
    std::string runtime_error;
    if (!impl_->tokenizer.load(path_to_utf8(root / "tokenizer" / "vocab.txt"),
                               path_to_utf8(root / "tokenizer" / "merges.txt"),
                               path_to_utf8(root / "tokenizer" / "special_tokens.json"),
                               &runtime_error))
    {
        return fail(error, "tokenizer_load", runtime_error);
    }

    std::string vision_param;
    std::string vision_bin;
    std::string pos_embed;
    if (resolve_dynamic_vision_paths(model_root, &vision_param, &vision_bin, &pos_embed))
    {
        impl_->dynamic_vision = impl_->make_vision_runtime();
        if (!impl_->dynamic_vision->load_dynamic(vision_param,
                                                 vision_bin,
                                                 pos_embed,
                                                 &runtime_error))
        {
            impl_->dynamic_vision.reset();
            return fail(error, "vision_load", runtime_error);
        }
    }

    impl_->text_runtime.reset(new TextRuntime(options.num_threads));
    if (!impl_->text_runtime->load(model_root, &runtime_error))
    {
        impl_->text_runtime.reset();
        impl_->dynamic_vision.reset();
        return fail(error, "text_load", runtime_error);
    }
    if (options.dflash && !impl_->text_runtime->load_dflash(model_root, &runtime_error))
    {
        impl_->text_runtime.reset();
        impl_->dynamic_vision.reset();
        return fail(error, "dflash_load", runtime_error);
    }

    impl_->ready = true;
    return true;
}

bool HunyuanOCR::infer_file(const std::string& image_path,
                            const InferenceRequest& request,
                            InferenceResult* result,
                            RuntimeError* error)
{
    if (result)
    {
        *result = InferenceResult();
    }
    clear_error(error);
    if (result == nullptr)
    {
        return fail(error, "argument", "inference result pointer is null");
    }
    if (!validate_request(request, error))
    {
        return false;
    }
    if (image_path.empty())
    {
        return fail(error, "image_input", "image path must not be empty");
    }
    if (!impl_ || !impl_->ready)
    {
        return fail(error, "runtime_state", "runtime is not loaded");
    }
    const auto total_start = std::chrono::steady_clock::now();
    const auto preprocess_start = total_start;
    ImagePreprocessResult image;
    std::string runtime_error;
    if (!impl_->preprocessor.preprocess_image_file(image_path, &image, &runtime_error))
    {
        return fail(error, "image_decode", runtime_error);
    }
    result->timing.preprocess_ms =
        elapsed_ms(preprocess_start, std::chrono::steady_clock::now());
    if (!impl_->infer_preprocessed(image, request, result, error))
    {
        return false;
    }
    result->timing.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    return true;
}

bool HunyuanOCR::infer_rgb(const std::vector<unsigned char>& rgb,
                           int width,
                           int height,
                           const InferenceRequest& request,
                           InferenceResult* result,
                           RuntimeError* error)
{
    if (result)
    {
        *result = InferenceResult();
    }
    clear_error(error);
    if (result == nullptr)
    {
        return fail(error, "argument", "inference result pointer is null");
    }
    if (!validate_request(request, error))
    {
        return false;
    }
    if (width <= 0 || height <= 0)
    {
        return fail(error, "image_input", "RGB dimensions must be positive");
    }
    const size_t image_width = static_cast<size_t>(width);
    const size_t image_height = static_cast<size_t>(height);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (image_width > max_size / image_height ||
        image_width * image_height > max_size / 3)
    {
        return fail(error, "image_input", "RGB dimensions overflow the byte count");
    }
    const size_t expected_size = image_width * image_height * 3;
    if (rgb.size() != expected_size)
    {
        return fail(error, "image_input", "RGB byte count does not match width and height");
    }
    if (!impl_ || !impl_->ready)
    {
        return fail(error, "runtime_state", "runtime is not loaded");
    }
    const auto total_start = std::chrono::steady_clock::now();
    const auto preprocess_start = total_start;
    ImagePreprocessResult image;
    std::string runtime_error;
    if (!impl_->preprocessor.preprocess_original_rgb(rgb, width, height, &image, &runtime_error))
    {
        return fail(error, "image_preprocess", runtime_error);
    }
    result->timing.preprocess_ms =
        elapsed_ms(preprocess_start, std::chrono::steady_clock::now());
    if (!impl_->infer_preprocessed(image, request, result, error))
    {
        return false;
    }
    result->timing.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    return true;
}

bool HunyuanOCR::ready() const
{
    return impl_ && impl_->ready;
}

const ModelLayoutReport& HunyuanOCR::layout_report() const
{
    static const ModelLayoutReport empty_report;
    return impl_ ? impl_->layout_report : empty_report;
}

} // namespace hunyuan_ocr
