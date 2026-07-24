#include "hunyuan_ocr/text_runtime.h"

#include "text_decoder_step.h"
#include "text_backend_policy.h"

#include "hunyuan_ocr/generation_config.h"
#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/multimodal_rope.h"
#include "hunyuan_ocr/precise_sdpa.h"
#include "hunyuan_ocr/utf8.h"
#include "mapped_model_file.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <sstream>
#include <unordered_set>

#if NCNN_VULKAN
#include <gpu.h>
#endif

namespace hunyuan_ocr {

bool text_vulkan_compiled()
{
#if NCNN_VULKAN
    return true;
#else
    return false;
#endif
}

namespace detail {

std::vector<float> build_mrope(const std::vector<int>& position_ids, int seq_len, bool use_cos)
{
    constexpr int head_dim = 128;
    constexpr int axis_count = 4;
    constexpr int section_size = head_dim / axis_count;
    const float base = static_cast<float>(10000.0 * std::pow(1000.0, 128.0 / 126.0));
    std::vector<float> values(static_cast<size_t>(seq_len) * head_dim);
    for (int dim = 0; dim < head_dim; ++dim)
    {
        const int axis = dim / section_size;
        const int frequency_dim = dim % (head_dim / 2);
        const float inv_freq = 1.0f / std::pow(base, static_cast<float>(2 * frequency_dim) / head_dim);
        for (int pos_index = 0; pos_index < seq_len; ++pos_index)
        {
            const int position = position_ids[static_cast<size_t>(axis) * seq_len + pos_index];
            const float angle = static_cast<float>(position) * inv_freq;
            values[static_cast<size_t>(pos_index) * head_dim + dim] =
                use_cos ? std::cos(angle) : std::sin(angle);
        }
    }
    return values;
}

bool extract_dflash_target_hidden(ncnn::Extractor& ex,
                                  std::array<ncnn::Mat, 4>* hidden,
                                  std::string* error)
{
    if (hidden == nullptr)
    {
        if (error) *error = "DFlash target hidden pointer is null";
        return false;
    }
    for (size_t index = 0; index < hidden->size(); ++index)
    {
        const std::string name = "out" + std::to_string(index + 1);
        if (ex.extract(name.c_str(), (*hidden)[index]) != 0)
        {
            if (error) *error = "extract failed: " + name;
            return false;
        }
    }
    return true;
}

} // namespace detail

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

constexpr int kHiddenSize = detail::kTextHiddenSize;
constexpr int kHeadDim = detail::kTextHeadDim;
constexpr int kAttentionLayerCount = detail::kTextAttentionLayerCount;
constexpr int kVocabSize = 120818;
constexpr float kRopeTheta = 10000.0f;
constexpr float kRopeAlpha = 1000.0f;
constexpr float kMaskNegInf = -1.0e38f;
constexpr float kDefaultRepetitionPenalty = 1.08f;

bool configure_net(ncnn::Net& net,
                   const std::filesystem::path& param,
                   int num_threads,
                   bool use_vulkan,
                   int vulkan_device,
                   std::string* error)
{
    (void)vulkan_device;
    net.opt = make_fp32_ncnn_option(num_threads);
    net.opt.use_packing_layout = false;
    if (use_vulkan)
    {
#if NCNN_VULKAN
        if (ncnn::create_gpu_instance() != 0)
        {
            if (error) *error = "failed to initialize ncnn Vulkan instance for text runtime";
            return false;
        }
        if (vulkan_device < 0 || vulkan_device >= ncnn::get_gpu_count())
        {
            if (error)
            {
                *error = "text Vulkan device index " + std::to_string(vulkan_device) +
                         " is out of range for " + std::to_string(ncnn::get_gpu_count()) +
                         " device(s)";
            }
            return false;
        }
        net.opt.use_vulkan_compute = true;
        net.opt.vulkan_device_index = vulkan_device;
#else
        if (error) *error = "text Vulkan requested, but ncnn was built without Vulkan support";
        return false;
#endif
    }

    if (net.load_param(param.c_str()) != 0)
    {
        if (error) *error = "failed to load param: " + path_to_utf8(param);
        return false;
    }
    return true;
}

bool load_net(ncnn::Net& net,
              const std::filesystem::path& param,
              const std::filesystem::path& bin,
              int num_threads,
              bool mmap_weights,
              bool use_vulkan,
              int vulkan_device,
              std::shared_ptr<MappedModelFile>* model_mapping,
              std::string* error)
{
    if (!configure_net(net, param, num_threads, use_vulkan, vulkan_device, error))
    {
        return false;
    }
    if (!load_model_file(net, bin, mmap_weights, model_mapping, error))
    {
        return false;
    }
    return true;
}

size_t logical_value_count(const ncnn::Mat& mat)
{
    const int h = mat.h > 0 ? mat.h : 1;
    const int d = mat.d > 0 ? mat.d : 1;
    const int c = mat.c > 0 ? mat.c : 1;
    const int elempack = mat.elempack > 0 ? mat.elempack : 1;
    return static_cast<size_t>(mat.w) * static_cast<size_t>(h) *
           static_cast<size_t>(d) * static_cast<size_t>(c) *
           static_cast<size_t>(elempack);
}

float rope_value(int position, int dim, bool use_cos)
{
    const float base = static_cast<float>(
        static_cast<double>(kRopeTheta) *
        std::pow(static_cast<double>(kRopeAlpha),
                 static_cast<double>(kHeadDim) / static_cast<double>(kHeadDim - 2)));
    const int half_dim = dim % (kHeadDim / 2);
    const float exponent = static_cast<float>(2 * half_dim) / static_cast<float>(kHeadDim);
    const float inv_freq = 1.0f / std::pow(base, exponent);
    const float angle = static_cast<float>(position) * inv_freq;
    return use_cos ? static_cast<float>(std::cos(angle)) : static_cast<float>(std::sin(angle));
}

ncnn::Mat make_mat_from_vector(int w, int h, const std::vector<float>& values)
{
    ncnn::Mat mat(w, h, const_cast<float*>(values.data()));
    return mat.clone();
}

ncnn::Mat make_mat_from_vector_3d(int w, int h, int c, const std::vector<float>& values)
{
    ncnn::Mat mat(w, h, c, const_cast<float*>(values.data()));
    return mat.clone();
}

ncnn::Mat make_prefill_mask(int seq_len)
{
    ncnn::Mat mask(seq_len, seq_len);
    for (int row_index = 0; row_index < seq_len; ++row_index)
    {
        float* row = mask.row(row_index);
        for (int col = 0; col < seq_len; ++col)
        {
            row[col] = col > row_index ? kMaskNegInf : 0.0f;
        }
    }
    return mask;
}

ncnn::Mat make_decode_mask(int past_len)
{
    ncnn::Mat mask(past_len + 1, 1);
    mask.fill(0.0f);
    return mask;
}

std::vector<float> build_decode_rope(int position, bool use_cos)
{
    std::vector<float> values(kHeadDim);
    for (int dim = 0; dim < kHeadDim; ++dim)
    {
        values[static_cast<size_t>(dim)] = rope_value(position, dim, use_cos);
    }
    return values;
}

std::vector<float> build_verify_rope(int start_position, bool use_cos)
{
    std::vector<float> values(
        static_cast<size_t>(detail::kDFlashBlockSize) * kHeadDim);
    for (int row = 0; row < detail::kDFlashBlockSize; ++row)
    {
        for (int dim = 0; dim < kHeadDim; ++dim)
        {
            values[static_cast<size_t>(row) * kHeadDim + dim] =
                rope_value(start_position + row, dim, use_cos);
        }
    }
    return values;
}

template <typename T>
bool read_binary_vector(const std::filesystem::path& path, size_t expected_count, std::vector<T>* values, std::string* error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        if (error) *error = "failed to open: " + path_to_utf8(path);
        return false;
    }
    values->assign(expected_count, T{});
    file.read(reinterpret_cast<char*>(values->data()), static_cast<std::streamsize>(expected_count * sizeof(T)));
    if (file.gcount() != static_cast<std::streamsize>(expected_count * sizeof(T)))
    {
        if (error) *error = "short read: " + path_to_utf8(path);
        return false;
    }
    char extra = 0;
    if (file.read(&extra, 1))
    {
        if (error) *error = "unexpected extra bytes: " + path_to_utf8(path);
        return false;
    }
    return true;
}

struct VlmFixtureMeta {
    int seq_len = 0;
    int expected_token_count = 0;
    int image_token_id = -1;
    int vision_token_count = 0;
};

bool parse_vlm_fixture_meta(const std::filesystem::path& path, VlmFixtureMeta* meta, std::string* error)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        if (error) *error = "failed to open fixture meta: " + path_to_utf8(path);
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        try
        {
            if (key == "seq_len") meta->seq_len = std::stoi(value);
            else if (key == "expected_token_count") meta->expected_token_count = std::stoi(value);
            else if (key == "image_token_id") meta->image_token_id = std::stoi(value);
            else if (key == "vision_token_count") meta->vision_token_count = std::stoi(value);
        }
        catch (const std::exception&)
        {
            if (error) *error = "invalid fixture meta value: " + line;
            return false;
        }
    }

    if (meta->seq_len <= 0 ||
        meta->expected_token_count <= 0 ||
        meta->image_token_id < 0 ||
        meta->vision_token_count <= 0)
    {
        if (error) *error = "vlm fixture meta must define positive seq_len, expected_token_count, image_token_id, and vision_token_count";
        return false;
    }
    return true;
}

struct VlmFixtureInputs {
    VlmFixtureMeta meta;
    std::vector<int> input_ids;
    std::vector<int> position_ids;
    std::vector<int> expected_tokens;
};

bool read_vlm_fixture_inputs(const std::filesystem::path& root,
                             VlmFixtureInputs* fixture,
                             std::string* error)
{
    if (fixture == nullptr)
    {
        if (error) *error = "VLM fixture output is null";
        return false;
    }
    VlmFixtureInputs loaded;
    if (!parse_vlm_fixture_meta(root / "meta.txt", &loaded.meta, error) ||
        !read_binary_vector(root / "input_ids.i32",
                            static_cast<size_t>(loaded.meta.seq_len),
                            &loaded.input_ids,
                            error) ||
        !read_binary_vector(root / "position_ids.i32",
                            static_cast<size_t>(loaded.meta.seq_len) * 4,
                            &loaded.position_ids,
                            error) ||
        !read_binary_vector(root / "expected_tokens.i32",
                            static_cast<size_t>(loaded.meta.expected_token_count),
                            &loaded.expected_tokens,
                            error))
    {
        return false;
    }
    *fixture = std::move(loaded);
    return true;
}

bool load_vlm_fixture_inputs(const std::string& fixture_dir,
                             VlmFixtureInputs* fixture,
                             std::string* error)
{
    return read_vlm_fixture_inputs(path_from_utf8(fixture_dir), fixture, error);
}

bool load_vlm_fixture_with_features(const std::string& fixture_dir,
                                    VlmFixtureInputs* fixture,
                                    std::vector<float>* vision_features,
                                    std::string* error)
{
    const std::filesystem::path root = path_from_utf8(fixture_dir);
    return read_vlm_fixture_inputs(root, fixture, error) &&
           read_binary_vector(root / "vision_features.f32",
                              static_cast<size_t>(fixture->meta.vision_token_count) *
                                  kHiddenSize,
                              vision_features,
                              error);
}

bool validate_external_vision_token_count(const VlmFixtureInputs& fixture,
                                          int vision_token_count,
                                          std::string* error)
{
    if (vision_token_count == fixture.meta.vision_token_count)
    {
        return true;
    }
    if (error) {
        *error = "external vision token count mismatch: got " +
                 std::to_string(vision_token_count) + ", expected " +
                 std::to_string(fixture.meta.vision_token_count);
    }
    return false;
}

bool validate_external_vision_feature_size(const VlmFixtureInputs& fixture,
                                           const std::vector<float>& vision_features,
                                           std::string* error)
{
    const size_t expected =
        static_cast<size_t>(fixture.meta.vision_token_count) * kHiddenSize;
    if (vision_features.size() == expected)
    {
        return true;
    }
    if (error) {
        *error = "external vision feature size mismatch: got " +
                 std::to_string(vision_features.size()) + ", expected " +
                 std::to_string(expected);
    }
    return false;
}

bool extract_or_error(ncnn::Extractor& ex, const char* name, ncnn::Mat* out, std::string* error)
{
    const int ret = ex.extract(name, *out);
    if (ret != 0)
    {
        if (error) *error = std::string("extract failed: ") + name;
        return false;
    }
    return true;
}

bool run_lm_head(const ncnn::Net& net, const ncnn::Mat& hidden, ncnn::Mat* logits, std::string* error)
{
    ncnn::Extractor ex = const_cast<ncnn::Net&>(net).create_extractor();
    if (ex.input("in0", hidden) != 0)
    {
        if (error) *error = "lm_head input failed";
        return false;
    }
    return extract_or_error(ex, "out0", logits, error);
}

bool run_text_embed(const ncnn::Net& net, int token_id, ncnn::Mat* embedding, std::string* error)
{
    ncnn::Mat token_mat(1, 1, 1);
    *static_cast<int*>(token_mat.data) = token_id;
    ncnn::Extractor ex = const_cast<ncnn::Net&>(net).create_extractor();
    if (ex.input("in0", token_mat) != 0)
    {
        if (error) *error = "text_embed input failed";
        return false;
    }
    return extract_or_error(ex, "out0", embedding, error);
}

bool run_text_embed_tokens(const ncnn::Net& net,
                           const std::vector<int>& input_ids,
                           int seq_len,
                           std::vector<float>* embeddings,
                           std::string* error)
{
    ncnn::Mat input_ids_mat(seq_len, 1);
    std::memcpy(input_ids_mat.data,
                input_ids.data(),
                static_cast<size_t>(seq_len) * sizeof(int));

    ncnn::Mat output;
    ncnn::Extractor ex = const_cast<ncnn::Net&>(net).create_extractor();
    if (ex.input("in0", input_ids_mat) != 0)
    {
        if (error) *error = "text_embed prompt input failed";
        return false;
    }
    if (!extract_or_error(ex, "out0", &output, error))
    {
        return false;
    }

    const size_t expected = static_cast<size_t>(seq_len) * kHiddenSize;
    if (logical_value_count(output) != expected)
    {
        if (error) {
            *error = "text_embed prompt output shape mismatch: got " +
                     std::to_string(logical_value_count(output)) +
                     ", expected " + std::to_string(expected);
        }
        return false;
    }

    embeddings->assign(expected, 0.0f);
    if (output.h == seq_len && output.w == kHiddenSize)
    {
        for (int row_index = 0; row_index < seq_len; ++row_index)
        {
            const float* src = output.row(row_index);
            std::memcpy(embeddings->data() + static_cast<size_t>(row_index) * kHiddenSize,
                        src,
                        static_cast<size_t>(kHiddenSize) * sizeof(float));
        }
        return true;
    }

    const float* src = output;
    std::memcpy(embeddings->data(), src, expected * sizeof(float));
    return true;
}

bool prepare_prompt_embeddings(const ncnn::Net& text_embed_net,
                               const std::vector<int>& input_ids,
                               const std::vector<int>& position_ids,
                               int image_token_id,
                               const std::vector<float>& vision_features,
                               int vision_token_count,
                               std::vector<float>* inputs_embeds,
                               std::string* error)
{
    if (inputs_embeds == nullptr)
    {
        if (error) *error = "prompt embedding output is null";
        return false;
    }
    const int seq_len = static_cast<int>(input_ids.size());
    if (seq_len <= 0)
    {
        if (error) *error = "prompt input_ids must not be empty";
        return false;
    }
    if (position_ids.size() != static_cast<size_t>(seq_len) * 4)
    {
        if (error) *error = "prompt position_ids size mismatch";
        return false;
    }
    if (vision_token_count <= 0 ||
        vision_features.size() != static_cast<size_t>(vision_token_count) * kHiddenSize)
    {
        if (error) *error = "vision feature size mismatch";
        return false;
    }
    if (!run_text_embed_tokens(text_embed_net, input_ids, seq_len, inputs_embeds, error))
    {
        return false;
    }

    int injected = 0;
    for (int index = 0; index < seq_len; ++index)
    {
        if (input_ids[static_cast<size_t>(index)] != image_token_id)
        {
            continue;
        }
        if (injected >= vision_token_count)
        {
            if (error) *error = "more image tokens than vision features";
            return false;
        }
        std::memcpy(inputs_embeds->data() + static_cast<size_t>(index) * kHiddenSize,
                    vision_features.data() + static_cast<size_t>(injected) * kHiddenSize,
                    static_cast<size_t>(kHiddenSize) * sizeof(float));
        ++injected;
    }
    if (injected != vision_token_count)
    {
        if (error) {
            *error = "image token count mismatch: injected " + std::to_string(injected) +
                     ", expected " + std::to_string(vision_token_count);
        }
        return false;
    }
    return true;
}

bool prepare_vlm_fixture_embeddings(const ncnn::Net& text_embed_net,
                                    const VlmFixtureInputs& fixture,
                                    const std::vector<float>& vision_features,
                                    std::vector<float>* inputs_embeds,
                                    std::string* error)
{
    return prepare_prompt_embeddings(text_embed_net,
                                     fixture.input_ids,
                                     fixture.position_ids,
                                     fixture.meta.image_token_id,
                                     vision_features,
                                     fixture.meta.vision_token_count,
                                     inputs_embeds,
                                     error);
}

using KVCache = detail::DecoderKVCache;

bool bind_decoder_inputs(ncnn::Extractor* ex,
                         const ncnn::Mat& input,
                         const ncnn::Mat& mask,
                         const ncnn::Mat& xd_cos,
                         const ncnn::Mat& xd_sin,
                         const ncnn::Mat& rope_cos,
                         const ncnn::Mat& rope_sin,
                         const char* error_message,
                         std::string* error)
{
    if (ex == nullptr ||
        ex->input("in0", input) != 0 ||
        ex->input("in1", mask) != 0 ||
        ex->input("in2", xd_cos) != 0 ||
        ex->input("in3", xd_sin) != 0 ||
        ex->input("in4", rope_cos) != 0 ||
        ex->input("in5", rope_sin) != 0)
    {
        if (error) *error = error_message;
        return false;
    }
    return true;
}

bool extract_decoder_outputs(ncnn::Extractor* ex,
                             ncnn::Mat* hidden,
                             KVCache* caches,
                             std::array<ncnn::Mat, 4>* target_hidden,
                             std::string* error)
{
    if (ex == nullptr || hidden == nullptr || caches == nullptr)
    {
        if (error) *error = "decoder output pointer is null";
        return false;
    }
    caches->clear();
    caches->reserve(kAttentionLayerCount);
    for (int layer = 0; layer < kAttentionLayerCount; ++layer)
    {
        char name_k[32];
        char name_v[32];
        std::snprintf(name_k, sizeof(name_k), "out_cache_k%d", layer);
        std::snprintf(name_v, sizeof(name_v), "out_cache_v%d", layer);
        ncnn::Mat key;
        ncnn::Mat value;
        if (!extract_or_error(*ex, name_k, &key, error) ||
            !extract_or_error(*ex, name_v, &value, error))
        {
            return false;
        }
        caches->emplace_back(std::move(key), std::move(value));
    }
    if (target_hidden != nullptr &&
        !detail::extract_dflash_target_hidden(*ex, target_hidden, error))
    {
        return false;
    }
    return extract_or_error(*ex, "out0", hidden, error);
}

bool run_decoder_prefill(const ncnn::Net& net,
                         const ncnn::Mat& inputs_embeds,
                         const ncnn::Mat& mask,
                         const ncnn::Mat& xd_cos,
                         const ncnn::Mat& xd_sin,
                         const ncnn::Mat& rope_cos,
                         const ncnn::Mat& rope_sin,
                         ncnn::Mat* hidden,
                         KVCache* caches,
                         std::array<ncnn::Mat, 4>* target_hidden,
                         std::string* error)
{
    ncnn::Extractor ex = const_cast<ncnn::Net&>(net).create_extractor();
    if (!bind_decoder_inputs(&ex,
                             inputs_embeds,
                             mask,
                             xd_cos,
                             xd_sin,
                             rope_cos,
                             rope_sin,
                             "decoder prefill input failed",
                             error))
    {
        return false;
    }
    return extract_decoder_outputs(&ex, hidden, caches, target_hidden, error);
}

// Production decoder step used by AR and DFlash verify. Defined in detail so
// microbenchmarks call the same forward path without forking protocol logic.
bool run_decoder_step_impl(const ncnn::Net& net,
                           const ncnn::Mat& current_embed,
                           const ncnn::Mat& mask,
                           const ncnn::Mat& cos,
                           const ncnn::Mat& sin,
                           const KVCache& caches,
                           ncnn::Mat* hidden,
                           KVCache* updated,
                           std::array<ncnn::Mat, 4>* target_hidden,
                           std::string* error)
{
    ncnn::Extractor ex = const_cast<ncnn::Net&>(net).create_extractor();
    if (!bind_decoder_inputs(&ex,
                             current_embed,
                             mask,
                             cos,
                             sin,
                             cos,
                             sin,
                             "decoder step input failed",
                             error))
    {
        return false;
    }

    if (static_cast<int>(caches.size()) != kAttentionLayerCount)
    {
        if (error) *error = "invalid KV cache layer count";
        return false;
    }

    for (int i = 0; i < kAttentionLayerCount; ++i)
    {
        char name_k[32];
        char name_v[32];
        std::snprintf(name_k, sizeof(name_k), "cache_k%d", i);
        std::snprintf(name_v, sizeof(name_v), "cache_v%d", i);
        if (ex.input(name_k, caches[static_cast<size_t>(i)].first) != 0 ||
            ex.input(name_v, caches[static_cast<size_t>(i)].second) != 0)
        {
            if (error)
            {
                *error = std::string("decoder step cache input failed at layer ") +
                         std::to_string(i);
            }
            return false;
        }
    }

    return extract_decoder_outputs(&ex, hidden, updated, target_hidden, error);
}

int select_next_token(const ncnn::Mat& logits, const std::vector<int>& history, float penalty, int* raw_top1)
{
    const size_t count = logical_value_count(logits);
    const float* src = logits;
    std::vector<float> scores(src, src + count);

    int raw_best = 0;
    for (size_t i = 1; i < count; ++i)
    {
        if (scores[i] > scores[static_cast<size_t>(raw_best)])
        {
            raw_best = static_cast<int>(i);
        }
    }
    if (raw_top1) *raw_top1 = raw_best;

    std::unordered_set<int> unique(history.begin(), history.end());
    for (const int token : unique)
    {
        if (token < 0 || token >= static_cast<int>(count)) continue;
        float& score = scores[static_cast<size_t>(token)];
        score = score < 0.0f ? score * penalty : score / penalty;
    }

    return static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
}

bool select_logit_rows(const ncnn::Mat& logits,
                       int row_count,
                       const std::vector<int>& initial_history,
                       const std::vector<int>* row_prefix,
                       float repetition_penalty,
                       std::vector<int>* tokens,
                       std::string* error)
{
    if (tokens == nullptr || row_count <= 0 || logits.w != kVocabSize || logits.h != row_count)
    {
        if (error)
        {
            *error = "lm_head row shape mismatch: got w=" + std::to_string(logits.w) +
                     ", h=" + std::to_string(logits.h) +
                     ", d=" + std::to_string(logits.d) +
                     ", c=" + std::to_string(logits.c) +
                     ", expected w=" + std::to_string(kVocabSize) +
                     ", h=" + std::to_string(row_count);
        }
        return false;
    }
    if (row_prefix != nullptr && row_prefix->size() < static_cast<size_t>(row_count))
    {
        if (error) *error = "lm_head row prefix is too short";
        return false;
    }

    tokens->clear();
    tokens->reserve(static_cast<size_t>(row_count));
    std::vector<int> history = initial_history;
    for (int row = 0; row < row_count; ++row)
    {
        if (row_prefix != nullptr)
        {
            history.push_back((*row_prefix)[static_cast<size_t>(row)]);
        }
        const ncnn::Mat row_logits = logits.row_range(row, 1);
        tokens->push_back(select_next_token(row_logits, history, repetition_penalty, nullptr));
    }
    return true;
}

struct DFlashPrefillState {
    KVCache caches;
    std::array<ncnn::Mat, 4> target_hidden;
    int first_token = -1;
};

struct DFlashBlockExecution {
    KVCache speculative_caches;
    std::array<ncnn::Mat, 4> speculative_target_hidden;
    std::vector<int> proposed_tokens;
    std::vector<int> target_tokens;
    int acceptance_length = -1;
    int correction_token = -1;
};

bool run_dflash_prefill(const ncnn::Net& decoder_net,
                        const ncnn::Net& lm_head_net,
                        int seq_len,
                        float repetition_penalty,
                        const std::vector<float>& inputs_embeds,
                        const std::vector<int>& input_ids,
                        const std::vector<int>& position_ids,
                        DFlashPrefillState* state,
                        DFlashDecodeTiming* timing,
                        std::string* error)
{
    const auto prefill_start = Clock::now();
    if (state == nullptr)
    {
        if (error) *error = "DFlash prefill state pointer is null";
        return false;
    }
    if (seq_len <= 0 ||
        inputs_embeds.size() != static_cast<size_t>(seq_len) * kHiddenSize ||
        input_ids.size() != static_cast<size_t>(seq_len) ||
        position_ids.size() != static_cast<size_t>(seq_len) * 4)
    {
        if (error) *error = "DFlash prefill input tensor sizes do not match metadata";
        return false;
    }

    const ncnn::Mat input_mat = make_mat_from_vector(kHiddenSize, seq_len, inputs_embeds);
    const ncnn::Mat prefill_mask = make_prefill_mask(seq_len);
    const std::vector<float> xd_cos_data = detail::build_mrope(position_ids, seq_len, true);
    const std::vector<float> xd_sin_data = detail::build_mrope(position_ids, seq_len, false);
    const ncnn::Mat xd_cos = make_mat_from_vector(kHeadDim, seq_len, xd_cos_data);
    const ncnn::Mat xd_sin = make_mat_from_vector(kHeadDim, seq_len, xd_sin_data);

    ncnn::Mat prefill_hidden;
    DFlashPrefillState local;
    if (!run_decoder_prefill(decoder_net,
                             input_mat,
                             prefill_mask,
                             xd_cos,
                             xd_sin,
                             xd_cos,
                             xd_sin,
                             &prefill_hidden,
                             &local.caches,
                             &local.target_hidden,
                             error))
    {
        return false;
    }

    ncnn::Mat first_logits;
    const ncnn::Mat last_hidden = prefill_hidden.row_range(seq_len - 1, 1);
    if (!run_lm_head(lm_head_net, last_hidden, &first_logits, error))
    {
        return false;
    }
    local.first_token =
        select_next_token(first_logits, input_ids, repetition_penalty, nullptr);
    *state = std::move(local);
    if (timing != nullptr)
    {
        timing->prefill_ms += elapsed_ms(prefill_start, Clock::now());
    }
    return true;
}

bool run_dflash_block(const ncnn::Net& text_embed_net,
                      const ncnn::Net& decoder_net,
                      const ncnn::Net& lm_head_net,
                      const DFlashDraftRuntime& draft_runtime,
                      int current_token,
                      float repetition_penalty,
                      const std::vector<int>& history_before_block,
                      const KVCache& caches,
                      const std::array<ncnn::Mat, 4>& target_hidden,
                      DFlashBlockExecution* execution,
                      DFlashDecodeTiming* timing,
                      std::string* error)
{
    if (execution == nullptr || caches.size() != kAttentionLayerCount)
    {
        if (error) *error = "invalid DFlash block state";
        return false;
    }
    const int context_len = target_hidden[0].h;
    if (current_token < 0 || context_len <= 0)
    {
        if (error) *error = "invalid DFlash block context";
        return false;
    }
    for (const ncnn::Mat& hidden : target_hidden)
    {
        if (hidden.dims != 2 || hidden.w != kHiddenSize || hidden.h != context_len)
        {
            if (error) *error = "DFlash target hidden shape mismatch";
            return false;
        }
    }

    const auto draft_prepare_start = Clock::now();
    std::vector<int> proposed = detail::build_dflash_block(current_token);

    std::vector<float> noise_embeddings;
    if (!run_text_embed_tokens(text_embed_net,
                               proposed,
                               detail::kDFlashBlockSize,
                               &noise_embeddings,
                               error))
    {
        return false;
    }

    const int draft_total_len = context_len + detail::kDFlashBlockSize;
    const std::vector<float> draft_cos_data = detail::build_dflash_rope(draft_total_len, true);
    const std::vector<float> draft_sin_data = detail::build_dflash_rope(draft_total_len, false);
    const std::vector<float> draft_mask_data = detail::build_dflash_attention_mask(context_len);
    DFlashDraftInput draft_input;
    draft_input.noise_embedding =
        make_mat_from_vector(kHiddenSize, detail::kDFlashBlockSize, noise_embeddings);
    draft_input.target_hidden = target_hidden;
    draft_input.cos = make_mat_from_vector_3d(kHeadDim, draft_total_len, 1, draft_cos_data);
    draft_input.sin = make_mat_from_vector_3d(kHeadDim, draft_total_len, 1, draft_sin_data);
    draft_input.attention_mask =
        make_mat_from_vector_3d(draft_total_len,
                                detail::kDFlashBlockSize,
                                1,
                                draft_mask_data);
    if (timing != nullptr)
    {
        timing->draft_prepare_ms += elapsed_ms(draft_prepare_start, Clock::now());
    }

    const auto draft_infer_start = Clock::now();
    ncnn::Mat draft_hidden;
    if (!draft_runtime.run(draft_input, &draft_hidden, error))
    {
        return false;
    }
    if (timing != nullptr)
    {
        timing->draft_infer_ms += elapsed_ms(draft_infer_start, Clock::now());
    }

    const auto draft_postprocess_start = Clock::now();
    ncnn::Mat draft_logits;
    const ncnn::Mat draft_prediction_hidden =
        draft_hidden.row_range(1, detail::kDFlashBlockSize - 1);
    if (!run_lm_head(lm_head_net, draft_prediction_hidden, &draft_logits, error))
    {
        return false;
    }
    std::vector<int> draft_tokens;
    if (!select_logit_rows(draft_logits,
                           detail::kDFlashBlockSize - 1,
                           {},
                           nullptr,
                           1.0f,
                           &draft_tokens,
                           error))
    {
        return false;
    }
    std::copy(draft_tokens.begin(), draft_tokens.end(), proposed.begin() + 1);
    if (timing != nullptr)
    {
        timing->draft_postprocess_ms += elapsed_ms(draft_postprocess_start, Clock::now());
    }

    const auto verify_prepare_start = Clock::now();
    std::vector<float> proposed_embeddings;
    if (!run_text_embed_tokens(text_embed_net,
                               proposed,
                               detail::kDFlashBlockSize,
                               &proposed_embeddings,
                               error))
    {
        return false;
    }
    const std::vector<float> verify_mask_data = detail::build_dflash_verify_mask(context_len);
    const std::vector<float> verify_cos_data = build_verify_rope(context_len, true);
    const std::vector<float> verify_sin_data = build_verify_rope(context_len, false);
    const ncnn::Mat proposed_mat =
        make_mat_from_vector(kHiddenSize, detail::kDFlashBlockSize, proposed_embeddings);
    const ncnn::Mat verify_mask =
        make_mat_from_vector(context_len + detail::kDFlashBlockSize,
                             detail::kDFlashBlockSize,
                             verify_mask_data);
    const ncnn::Mat verify_cos =
        make_mat_from_vector(kHeadDim, detail::kDFlashBlockSize, verify_cos_data);
    const ncnn::Mat verify_sin =
        make_mat_from_vector(kHeadDim, detail::kDFlashBlockSize, verify_sin_data);
    if (timing != nullptr)
    {
        timing->verify_prepare_ms += elapsed_ms(verify_prepare_start, Clock::now());
    }

    const auto verify_infer_start = Clock::now();
    ncnn::Mat verify_hidden;
    KVCache speculative_caches;
    std::array<ncnn::Mat, 4> speculative_target_hidden;
    if (!run_decoder_step_impl(decoder_net,
                          proposed_mat,
                          verify_mask,
                          verify_cos,
                          verify_sin,
                          caches,
                          &verify_hidden,
                          &speculative_caches,
                          &speculative_target_hidden,
                          error))
    {
        return false;
    }
    if (timing != nullptr)
    {
        timing->verify_infer_ms += elapsed_ms(verify_infer_start, Clock::now());
    }

    const auto verify_postprocess_start = Clock::now();
    ncnn::Mat target_logits;
    if (!run_lm_head(lm_head_net, verify_hidden, &target_logits, error))
    {
        return false;
    }
    std::vector<int> posterior;
    if (!select_logit_rows(target_logits,
                           detail::kDFlashBlockSize,
                           history_before_block,
                           &proposed,
                           repetition_penalty,
                           &posterior,
                           error))
    {
        return false;
    }

    const int acceptance_length = detail::dflash_acceptance_length(proposed, posterior);
    if (acceptance_length < 0 || acceptance_length >= detail::kDFlashBlockSize)
    {
        if (error) *error = "invalid DFlash acceptance length";
        return false;
    }

    DFlashBlockExecution local;
    local.speculative_caches = std::move(speculative_caches);
    local.speculative_target_hidden = std::move(speculative_target_hidden);
    local.proposed_tokens = std::move(proposed);
    local.target_tokens = std::move(posterior);
    local.acceptance_length = acceptance_length;
    local.correction_token =
        local.target_tokens[static_cast<size_t>(acceptance_length)];
    *execution = std::move(local);
    if (timing != nullptr)
    {
        timing->verify_postprocess_ms += elapsed_ms(verify_postprocess_start, Clock::now());
    }
    return true;
}

bool run_dflash_block_probe(const ncnn::Net& text_embed_net,
                            const ncnn::Net& decoder_net,
                            const ncnn::Net& lm_head_net,
                            const DFlashDraftRuntime& draft_runtime,
                            int seq_len,
                            float repetition_penalty,
                            const std::vector<float>& inputs_embeds,
                            const std::vector<int>& input_ids,
                            const std::vector<int>& position_ids,
                            const std::vector<int>& expected_tokens,
                            DFlashBlockProbeResult* result,
                            std::string* error)
{
    if (result == nullptr)
    {
        if (error) *error = "DFlash probe result pointer is null";
        return false;
    }
    if (expected_tokens.size() < 2)
    {
        if (error) *error = "DFlash probe requires two expected tokens";
        return false;
    }

    DFlashPrefillState prefill;
    if (!run_dflash_prefill(decoder_net,
                            lm_head_net,
                            seq_len,
                            repetition_penalty,
                            inputs_embeds,
                            input_ids,
                            position_ids,
                            &prefill,
                            nullptr,
                            error))
    {
        return false;
    }

    DFlashBlockExecution block;
    if (!run_dflash_block(text_embed_net,
                          decoder_net,
                          lm_head_net,
                          draft_runtime,
                          prefill.first_token,
                          repetition_penalty,
                          input_ids,
                          prefill.caches,
                          prefill.target_hidden,
                          &block,
                          nullptr,
                          error))
    {
        return false;
    }

    DFlashBlockProbeResult local;
    local.seq_len = seq_len;
    local.first_token = prefill.first_token;
    local.acceptance_length = block.acceptance_length;
    local.correction_token = block.correction_token;
    local.first_token_matches_expected = prefill.first_token == expected_tokens[0];
    local.first_target_token_matches_expected = block.target_tokens[0] == expected_tokens[1];
    local.proposed_tokens = std::move(block.proposed_tokens);
    local.target_tokens = std::move(block.target_tokens);
    *result = std::move(local);
    return true;
}

bool decode_dflash_from_embeddings(const ncnn::Net& text_embed_net,
                                   const ncnn::Net& decoder_net,
                                   const ncnn::Net& lm_head_net,
                                   const DFlashDraftRuntime& draft_runtime,
                                   int seq_len,
                                   int max_tokens,
                                   float repetition_penalty,
                                   const std::vector<float>& inputs_embeds,
                                   const std::vector<int>& input_ids,
                                   const std::vector<int>& position_ids,
                                   const std::vector<int>& expected_tokens,
                                   const std::vector<int>& eos_ids,
                                   DFlashDecodeResult* result,
                                   std::string* error,
                                   const TextTokenCallback& token_callback = {})
{
    if (result == nullptr)
    {
        if (error) *error = "DFlash decode result pointer is null";
        return false;
    }
    if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f)
    {
        if (error) *error = "repetition_penalty must be positive";
        return false;
    }
    if (expected_tokens.empty())
    {
        if (max_tokens <= 0) max_tokens = 1024;
    }
    else if (max_tokens <= 0 || max_tokens > static_cast<int>(expected_tokens.size()))
    {
        max_tokens = static_cast<int>(expected_tokens.size());
    }

    const auto total_start = Clock::now();
    DFlashDecodeResult local;
    local.decode.seq_len = seq_len;
    local.decode.checked_tokens = max_tokens;
    local.decode.repetition_penalty = repetition_penalty;
    if (!expected_tokens.empty())
    {
        local.decode.expected_tokens.assign(expected_tokens.begin(),
                                            expected_tokens.begin() + max_tokens);
    }

    DFlashPrefillState state;
    if (!run_dflash_prefill(decoder_net,
                            lm_head_net,
                            seq_len,
                            repetition_penalty,
                            inputs_embeds,
                            input_ids,
                            position_ids,
                            &state,
                            &local.timing,
                            error))
    {
        return false;
    }

    std::vector<int> history_before_block = input_ids;
    int current_token = state.first_token;
    bool stop = false;
    while (static_cast<int>(local.decode.generated_tokens.size()) < max_tokens && !stop)
    {
        const int context_len = state.target_hidden[0].h;
        DFlashBlockExecution block;
        if (!run_dflash_block(text_embed_net,
                              decoder_net,
                              lm_head_net,
                              draft_runtime,
                              current_token,
                              repetition_penalty,
                              history_before_block,
                              state.caches,
                              state.target_hidden,
                              &block,
                              &local.timing,
                              error))
        {
            return false;
        }

        const auto commit_start = Clock::now();
        const int available = block.acceptance_length + 1;
        const int remaining = max_tokens -
            static_cast<int>(local.decode.generated_tokens.size());
        int committed = std::min(available, remaining);
        for (int index = 0; index < committed; ++index)
        {
            const int token = block.proposed_tokens[static_cast<size_t>(index)];
            local.decode.generated_tokens.push_back(token);
            if (token_callback)
            {
                token_callback(token);
            }
            if (is_eos_token(token, eos_ids))
            {
                committed = index + 1;
                stop = true;
                break;
            }
        }

        const int accepted_draft_tokens = std::max(0, committed - 1);
        ++local.block_count;
        local.drafted_token_count += detail::kDFlashBlockSize - 1;
        local.accepted_draft_token_count += accepted_draft_tokens;
        local.acceptance_lengths.push_back(accepted_draft_tokens);

        if (committed < available || stop ||
            static_cast<int>(local.decode.generated_tokens.size()) >= max_tokens)
        {
            local.timing.commit_ms += elapsed_ms(commit_start, Clock::now());
            break;
        }

        KVCache committed_caches;
        committed_caches.reserve(state.caches.size());
        const int committed_context_len = context_len + committed;
        for (size_t layer = 0; layer < block.speculative_caches.size(); ++layer)
        {
            ncnn::Mat key;
            ncnn::Mat value;
            if (!detail::view_dflash_rows(block.speculative_caches[layer].first,
                                          committed_context_len,
                                          &key,
                                          error) ||
                !detail::view_dflash_rows(block.speculative_caches[layer].second,
                                          committed_context_len,
                                          &value,
                                          error))
            {
                return false;
            }
            committed_caches.emplace_back(std::move(key), std::move(value));
        }
        state.caches = std::move(committed_caches);
        for (size_t index = 0; index < state.target_hidden.size(); ++index)
        {
            if (!detail::append_dflash_rows(&state.target_hidden[index],
                                            block.speculative_target_hidden[index],
                                            committed,
                                            error))
            {
                return false;
            }
        }
        history_before_block.insert(history_before_block.end(),
                                    block.proposed_tokens.begin(),
                                    block.proposed_tokens.begin() + committed);
        current_token = block.correction_token;
        local.timing.commit_ms += elapsed_ms(commit_start, Clock::now());
    }

    if (!local.decode.expected_tokens.empty())
    {
        local.decode.expected_tokens.resize(local.decode.generated_tokens.size());
    }
    local.timing.total_ms = elapsed_ms(total_start, Clock::now());
    *result = std::move(local);
    return true;
}

bool decode_from_embeddings(const ncnn::Net& text_embed_net,
                            const ncnn::Net& decoder_net,
                            const ncnn::Net& lm_head_net,
                            int seq_len,
                            int max_tokens,
                            float repetition_penalty,
                            const std::vector<float>& inputs_embeds,
                            const std::vector<int>& input_ids,
                            const std::vector<int>& position_ids,
                            const std::vector<int>& expected_tokens,
                            const std::vector<int>& eos_ids,
                            TextDecodeResult* result,
                            TextDecodeTiming* timing,
                            std::string* error,
                            const TextTokenCallback& token_callback = {})
{
    const auto total_start = Clock::now();
    if (seq_len <= 0)
    {
        if (error) *error = "seq_len must be positive";
        return false;
    }
    if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f)
    {
        if (error) *error = "repetition_penalty must be positive";
        return false;
    }
    if (inputs_embeds.size() != static_cast<size_t>(seq_len) * kHiddenSize ||
        input_ids.size() != static_cast<size_t>(seq_len) ||
        position_ids.size() != static_cast<size_t>(seq_len) * 4)
    {
        if (error) *error = "decode input tensor sizes do not match fixture metadata";
        return false;
    }
    if (expected_tokens.empty())
    {
        if (max_tokens <= 0)
        {
            max_tokens = 1024;
        }
    }
    else if (max_tokens <= 0 || max_tokens > static_cast<int>(expected_tokens.size()))
    {
        max_tokens = static_cast<int>(expected_tokens.size());
    }

    const ncnn::Mat input_mat = make_mat_from_vector(kHiddenSize, seq_len, inputs_embeds);
    const ncnn::Mat prefill_mask = make_prefill_mask(seq_len);
    const std::vector<float> xd_cos_data = detail::build_mrope(position_ids, seq_len, true);
    const std::vector<float> xd_sin_data = detail::build_mrope(position_ids, seq_len, false);
    const ncnn::Mat xd_cos = make_mat_from_vector(kHeadDim, seq_len, xd_cos_data);
    const ncnn::Mat xd_sin = make_mat_from_vector(kHeadDim, seq_len, xd_sin_data);

    ncnn::Mat prefill_hidden;
    KVCache caches;
    const auto prefill_start = Clock::now();
    if (!run_decoder_prefill(decoder_net, input_mat, prefill_mask, xd_cos, xd_sin, xd_cos, xd_sin,
                             &prefill_hidden, &caches, nullptr, error))
    {
        return false;
    }
#if NCNN_VULKAN
    std::unique_ptr<detail::VulkanDecoderSession> vulkan_decoder_session;
    if (decoder_net.opt.use_vulkan_compute)
    {
        vulkan_decoder_session =
            std::make_unique<detail::VulkanDecoderSession>(decoder_net, lm_head_net);
        if (!vulkan_decoder_session->initialize(caches, error))
        {
            return false;
        }
        caches.clear();
    }
#endif
    const auto prefill_end = Clock::now();
    if (timing)
    {
        timing->prefill_ms += elapsed_ms(prefill_start, prefill_end);
    }

    ncnn::Mat logits;
    ncnn::Mat last_hidden = prefill_hidden.row_range(seq_len - 1, 1).clone();
    const auto first_lm_head_start = Clock::now();
    if (!run_lm_head(lm_head_net, last_hidden, &logits, error))
    {
        return false;
    }
    if (timing)
    {
        timing->lm_head_ms += elapsed_ms(first_lm_head_start, Clock::now());
    }

    TextDecodeResult local;
    local.seq_len = seq_len;
    local.repetition_penalty = repetition_penalty;
    local.checked_tokens = max_tokens;
    if (!expected_tokens.empty())
    {
        local.expected_tokens.assign(expected_tokens.begin(), expected_tokens.begin() + max_tokens);
    }

    std::vector<int> history = input_ids;
    int raw_top1 = -1;
    const auto first_token_select_start = Clock::now();
    int current_token = select_next_token(logits, history, local.repetition_penalty, &raw_top1);
    if (timing)
    {
        timing->token_select_ms += elapsed_ms(first_token_select_start, Clock::now());
    }

    for (int out_index = 0; out_index < max_tokens; ++out_index)
    {
        local.generated_tokens.push_back(current_token);
        local.raw_top1_tokens.push_back(raw_top1);
        history.push_back(current_token);
        if (token_callback)
        {
            token_callback(current_token);
        }

        if (is_eos_token(current_token, eos_ids) || out_index + 1 >= max_tokens)
        {
            break;
        }

        ncnn::Mat current_embed;
        const auto decode_text_embed_start = Clock::now();
        if (!run_text_embed(text_embed_net, current_token, &current_embed, error))
        {
            return false;
        }
        if (timing)
        {
            timing->text_embed_ms += elapsed_ms(decode_text_embed_start, Clock::now());
        }

        const auto decoder_step_start = Clock::now();
        const int position_value = seq_len + out_index;
        const ncnn::Mat decode_mask = make_decode_mask(position_value);
        const std::vector<float> decode_cos_data = build_decode_rope(position_value, true);
        const std::vector<float> decode_sin_data = build_decode_rope(position_value, false);
        ncnn::Mat decode_cos(kHeadDim, const_cast<float*>(decode_cos_data.data()));
        ncnn::Mat decode_sin(kHeadDim, const_cast<float*>(decode_sin_data.data()));
        decode_cos = decode_cos.clone();
        decode_sin = decode_sin.clone();

        ncnn::Mat decode_hidden;
        bool logits_ready = false;
#if NCNN_VULKAN
        if (vulkan_decoder_session)
        {
            if (!vulkan_decoder_session->run_step_logits(current_embed,
                                                         decode_mask,
                                                         decode_cos,
                                                         decode_sin,
                                                         &logits,
                                                         error))
            {
                return false;
            }
            logits_ready = true;
        }
        else
#endif
        {
            KVCache updated;
            if (!run_decoder_step_impl(decoder_net,
                                       current_embed,
                                       decode_mask,
                                       decode_cos,
                                       decode_sin,
                                       caches,
                                       &decode_hidden,
                                       &updated,
                                       nullptr,
                                       error))
            {
                return false;
            }
            caches = std::move(updated);
        }
        if (timing)
        {
            timing->decode_ms += elapsed_ms(decoder_step_start, Clock::now());
        }

        const auto lm_head_start = Clock::now();
        if (!logits_ready && !run_lm_head(lm_head_net, decode_hidden, &logits, error))
        {
            return false;
        }
        if (timing && !logits_ready)
        {
            timing->lm_head_ms += elapsed_ms(lm_head_start, Clock::now());
        }
        const auto token_select_start = Clock::now();
        current_token = select_next_token(logits, history, local.repetition_penalty, &raw_top1);
        if (timing)
        {
            timing->token_select_ms += elapsed_ms(token_select_start, Clock::now());
        }
    }
    if (timing)
    {
        timing->total_ms += elapsed_ms(total_start, Clock::now());
        local.timing = *timing;
    }

    if (!local.expected_tokens.empty())
    {
        local.expected_tokens.resize(local.generated_tokens.size());
    }
    *result = std::move(local);
    return true;
}

bool decode_from_prompt_tokens(const ncnn::Net& text_embed_net,
                               const ncnn::Net& decoder_net,
                               const ncnn::Net& lm_head_net,
                               const std::vector<int>& input_ids,
                               const std::vector<int>& position_ids,
                               int image_token_id,
                               const std::vector<float>& vision_features,
                               int vision_token_count,
                               const std::vector<int>& expected_tokens,
                               const std::vector<int>& eos_ids,
                               int max_tokens,
                               float repetition_penalty,
                               TextDecodeResult* result,
                               TextDecodeTiming* timing,
                               std::string* error,
                               const TextTokenCallback& token_callback)
{
    const int seq_len = static_cast<int>(input_ids.size());
    std::vector<float> inputs_embeds;
    const auto text_embed_start = Clock::now();
    if (!prepare_prompt_embeddings(text_embed_net,
                                   input_ids,
                                   position_ids,
                                   image_token_id,
                                   vision_features,
                                   vision_token_count,
                                   &inputs_embeds,
                                   error))
    {
        return false;
    }
    if (timing)
    {
        timing->text_embed_ms += elapsed_ms(text_embed_start, Clock::now());
    }

    return decode_from_embeddings(text_embed_net,
                                  decoder_net,
                                  lm_head_net,
                                  seq_len,
                                  max_tokens,
                                  repetition_penalty,
                                  inputs_embeds,
                                  input_ids,
                                  position_ids,
                                  expected_tokens,
                                  eos_ids,
                                  result,
                                  timing,
                                  error,
                                  token_callback);
}

bool decode_dflash_from_prompt_tokens(const ncnn::Net& text_embed_net,
                                      const ncnn::Net& decoder_net,
                                      const ncnn::Net& lm_head_net,
                                      const DFlashDraftRuntime& draft_runtime,
                                      const std::vector<int>& input_ids,
                                      const std::vector<int>& position_ids,
                                      int image_token_id,
                                      const std::vector<float>& vision_features,
                                      int vision_token_count,
                                      const std::vector<int>& expected_tokens,
                                      const std::vector<int>& eos_ids,
                                      int max_tokens,
                                      float repetition_penalty,
                                      DFlashDecodeResult* result,
                                      std::string* error,
                                      const TextTokenCallback& token_callback)
{
    const int seq_len = static_cast<int>(input_ids.size());
    std::vector<float> inputs_embeds;
    if (!prepare_prompt_embeddings(text_embed_net,
                                   input_ids,
                                   position_ids,
                                   image_token_id,
                                   vision_features,
                                   vision_token_count,
                                   &inputs_embeds,
                                   error))
    {
        return false;
    }

    return decode_dflash_from_embeddings(text_embed_net,
                                         decoder_net,
                                         lm_head_net,
                                         draft_runtime,
                                         seq_len,
                                         max_tokens,
                                         repetition_penalty,
                                         inputs_embeds,
                                         input_ids,
                                         position_ids,
                                         expected_tokens,
                                         eos_ids,
                                         result,
                                         error,
                                         token_callback);
}

template <typename Result>
bool validate_text_runtime_request(bool runtime_ready,
                                   const Result* result,
                                   const char* null_result_error,
                                   std::string* error)
{
    if (!runtime_ready)
    {
        if (error) *error = "text runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = null_result_error;
        return false;
    }
    return true;
}

template <typename Result>
bool validate_dflash_runtime_request(bool runtime_ready,
                                     bool dflash_runtime_ready,
                                     const Result* result,
                                     const char* null_result_error,
                                     std::string* error)
{
    if (!runtime_ready)
    {
        if (error) *error = "text runtime is not loaded";
        return false;
    }
    if (!dflash_runtime_ready)
    {
        if (error) *error = "DFlash draft runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = null_result_error;
        return false;
    }
    return true;
}

} // namespace

TextRuntime::TextRuntime(int num_threads,
                         bool mmap_weights,
                         bool use_vulkan,
                         int vulkan_device)
    : text_embed_net_(new ncnn::Net),
      text_decoder_net_(new ncnn::Net),
      lm_head_net_(new ncnn::Net),
      dflash_draft_(new DFlashDraftRuntime(num_threads, mmap_weights)),
      num_threads_(num_threads),
      mmap_weights_(mmap_weights),
      use_vulkan_(use_vulkan),
      vulkan_device_(vulkan_device)
{
}

bool TextRuntime::load(const std::string& model_root, std::string* error)
{
    const std::filesystem::path root = path_from_utf8(model_root);
    ready_ = false;
    dflash_draft_.reset(new DFlashDraftRuntime(num_threads_, mmap_weights_));
    text_embed_net_.reset();
    text_decoder_net_.reset();
    lm_head_net_.reset();
    text_decoder_model_mapping_.reset();
    tied_model_data_.reset();
    text_embed_net_.reset(new ncnn::Net);
    text_decoder_net_.reset(new ncnn::Net);
    lm_head_net_.reset(new ncnn::Net);
    eos_ids_.clear();

    if (!load_eos_ids(path_to_utf8(root / "tokenizer" / "eos_ids.json"), &eos_ids_, error))
    {
        return false;
    }

    tied_model_data_.reset(new SharedModelData);
    if (!tied_model_data_->open(root / "text_embed" / "text_embed.ncnn.bin",
                                mmap_weights_,
                                error))
    {
        return false;
    }
    if (!configure_net(*text_embed_net_,
                       root / "text_embed" / "text_embed.ncnn.param",
                       num_threads_,
                       detail::text_stage_uses_vulkan(detail::TextNetStage::Embedding,
                                                       use_vulkan_),
                       vulkan_device_,
                       error) ||
        !load_shared_model_data(*text_embed_net_, *tied_model_data_, error))
    {
        return false;
    }
    if (!detail::load_text_decoder_kv_net(model_root,
                                          num_threads_,
                                          text_decoder_net_.get(),
                                          error,
                                          mmap_weights_,
                                          &text_decoder_model_mapping_,
                                          detail::text_stage_uses_vulkan(detail::TextNetStage::Decoder,
                                                                          use_vulkan_),
                                          vulkan_device_))
    {
        return false;
    }
    if (!configure_net(*lm_head_net_,
                       root / "lm_head" / "lm_head.ncnn.param",
                       num_threads_,
                       detail::text_stage_uses_vulkan(detail::TextNetStage::LmHead,
                                                       use_vulkan_),
                       vulkan_device_,
                       error) ||
        !load_shared_model_data(*lm_head_net_, *tied_model_data_, error))
    {
        return false;
    }

    ready_ = true;
    return true;
}

bool TextRuntime::load_dflash(const std::string& model_root, std::string* error)
{
    if (!ready_)
    {
        if (error) *error = "text runtime is not loaded";
        return false;
    }
    if (use_vulkan_)
    {
        if (error) *error = "DFlash is not supported with Text Vulkan yet";
        return false;
    }
    const std::filesystem::path root = path_from_utf8(model_root);
    return dflash_draft_->load(path_to_utf8(root / "dflash" / "dflash.ncnn.param"),
                               path_to_utf8(root / "dflash" / "dflash.ncnn.bin"),
                               error);
}

bool TextRuntime::ready() const
{
    return ready_;
}

bool TextRuntime::dflash_ready() const
{
    return dflash_draft_ != nullptr && dflash_draft_->ready();
}

size_t TextRuntime::mapped_weight_bytes() const
{
    size_t bytes = 0;
    if (tied_model_data_) bytes += tied_model_data_->mapped_bytes();
    if (text_decoder_model_mapping_) bytes += text_decoder_model_mapping_->size();
    if (dflash_draft_) bytes += dflash_draft_->mapped_weight_bytes();
    return bytes;
}

bool TextDecodeResult::matches_expected() const
{
    return generated_tokens == expected_tokens;
}

bool TextRuntime::run_vlm_fixture_decode(const std::string& fixture_dir,
                                         int max_tokens,
                                         TextDecodeResult* result,
                                         std::string* error) const
{
    if (!validate_text_runtime_request(ready_, result, "result pointer is null", error))
    {
        return false;
    }

    VlmFixtureInputs fixture;
    std::vector<float> vision_features;
    if (!load_vlm_fixture_with_features(fixture_dir,
                                        &fixture,
                                        &vision_features,
                                        error))
    {
        return false;
    }

    std::vector<float> inputs_embeds;
    if (!prepare_vlm_fixture_embeddings(*text_embed_net_,
                                        fixture,
                                        vision_features,
                                        &inputs_embeds,
                                        error))
    {
        return false;
    }

    return decode_from_embeddings(*text_embed_net_,
                                  *text_decoder_net_,
                                  *lm_head_net_,
                                  fixture.meta.seq_len,
                                  max_tokens,
                                  kDefaultRepetitionPenalty,
                                  inputs_embeds,
                                  fixture.input_ids,
                                  fixture.position_ids,
                                  fixture.expected_tokens,
                                  eos_ids_,
                                  result,
                                  &result->timing,
                                  error);
}

bool TextRuntime::run_vlm_fixture_dflash_probe(const std::string& fixture_dir,
                                               DFlashBlockProbeResult* result,
                                               std::string* error) const
{
    if (!validate_dflash_runtime_request(ready_,
                                         dflash_ready(),
                                         result,
                                         "DFlash probe result pointer is null",
                                         error))
    {
        return false;
    }

    VlmFixtureInputs fixture;
    std::vector<float> vision_features;
    if (!load_vlm_fixture_with_features(fixture_dir,
                                        &fixture,
                                        &vision_features,
                                        error))
    {
        return false;
    }

    std::vector<float> inputs_embeds;
    if (!prepare_vlm_fixture_embeddings(*text_embed_net_,
                                        fixture,
                                        vision_features,
                                        &inputs_embeds,
                                        error))
    {
        return false;
    }

    return run_dflash_block_probe(*text_embed_net_,
                                  *text_decoder_net_,
                                  *lm_head_net_,
                                  *dflash_draft_,
                                  fixture.meta.seq_len,
                                  kDefaultRepetitionPenalty,
                                  inputs_embeds,
                                  fixture.input_ids,
                                  fixture.position_ids,
                                  fixture.expected_tokens,
                                  result,
                                  error);
}

bool TextRuntime::run_vlm_fixture_dflash_decode(const std::string& fixture_dir,
                                                int max_tokens,
                                                DFlashDecodeResult* result,
                                                std::string* error) const
{
    if (!validate_dflash_runtime_request(ready_,
                                         dflash_ready(),
                                         result,
                                         "DFlash decode result pointer is null",
                                         error))
    {
        return false;
    }

    VlmFixtureInputs fixture;
    std::vector<float> vision_features;
    if (!load_vlm_fixture_with_features(fixture_dir,
                                        &fixture,
                                        &vision_features,
                                        error))
    {
        return false;
    }

    return run_vlm_dflash_decode_with_prompt(fixture.input_ids,
                                             fixture.position_ids,
                                             fixture.meta.image_token_id,
                                             vision_features,
                                             fixture.meta.vision_token_count,
                                             fixture.expected_tokens,
                                             max_tokens,
                                             kDefaultRepetitionPenalty,
                                             result,
                                             error);
}

bool TextRuntime::run_vlm_fixture_dflash_decode_with_features(
    const std::string& fixture_dir,
    const std::vector<float>& vision_features,
    int vision_token_count,
    int max_tokens,
    DFlashDecodeResult* result,
    std::string* error) const
{
    if (!validate_dflash_runtime_request(ready_,
                                         dflash_ready(),
                                         result,
                                         "DFlash decode result pointer is null",
                                         error))
    {
        return false;
    }

    VlmFixtureInputs fixture;
    if (!load_vlm_fixture_inputs(fixture_dir, &fixture, error))
    {
        return false;
    }
    if (!validate_external_vision_token_count(fixture, vision_token_count, error))
    {
        return false;
    }

    return run_vlm_dflash_decode_with_prompt(fixture.input_ids,
                                             fixture.position_ids,
                                             fixture.meta.image_token_id,
                                             vision_features,
                                             vision_token_count,
                                             fixture.expected_tokens,
                                             max_tokens,
                                             kDefaultRepetitionPenalty,
                                             result,
                                             error);
}

bool TextRuntime::run_vlm_dflash_decode_with_prompt(
    const std::vector<int>& input_ids,
    const std::vector<int>& position_ids,
    int image_token_id,
    const std::vector<float>& vision_features,
    int vision_token_count,
    const std::vector<int>& expected_tokens,
    int max_tokens,
    float repetition_penalty,
    DFlashDecodeResult* result,
    std::string* error,
    const TextTokenCallback& token_callback) const
{
    if (!validate_dflash_runtime_request(ready_,
                                         dflash_ready(),
                                         result,
                                         "DFlash decode result pointer is null",
                                         error))
    {
        return false;
    }

    return decode_dflash_from_prompt_tokens(*text_embed_net_,
                                            *text_decoder_net_,
                                            *lm_head_net_,
                                            *dflash_draft_,
                                            input_ids,
                                            position_ids,
                                            image_token_id,
                                            vision_features,
                                            vision_token_count,
                                            expected_tokens,
                                            eos_ids_,
                                            max_tokens,
                                            repetition_penalty,
                                            result,
                                            error,
                                            token_callback);
}

bool TextRuntime::run_vlm_fixture_decode_with_features(const std::string& fixture_dir,
                                                       const std::vector<float>& vision_features,
                                                       int vision_token_count,
                                                       int max_tokens,
                                                       TextDecodeResult* result,
                                                       std::string* error) const
{
    if (!validate_text_runtime_request(ready_, result, "result pointer is null", error))
    {
        return false;
    }

    VlmFixtureInputs fixture;
    if (!load_vlm_fixture_inputs(fixture_dir, &fixture, error))
    {
        return false;
    }
    if (!validate_external_vision_token_count(fixture, vision_token_count, error))
    {
        return false;
    }
    if (!validate_external_vision_feature_size(fixture, vision_features, error))
    {
        return false;
    }

    std::vector<float> inputs_embeds;
    if (!prepare_vlm_fixture_embeddings(*text_embed_net_,
                                        fixture,
                                        vision_features,
                                        &inputs_embeds,
                                        error))
    {
        return false;
    }

    return decode_from_embeddings(*text_embed_net_,
                                  *text_decoder_net_,
                                  *lm_head_net_,
                                  fixture.meta.seq_len,
                                  max_tokens,
                                  kDefaultRepetitionPenalty,
                                  inputs_embeds,
                                  fixture.input_ids,
                                  fixture.position_ids,
                                  fixture.expected_tokens,
                                  eos_ids_,
                                  result,
                                  nullptr,
                                  error);
}

bool TextRuntime::run_vlm_decode_with_prompt(const std::vector<int>& input_ids,
                                             const std::vector<int>& position_ids,
                                             int image_token_id,
                                             const std::vector<float>& vision_features,
                                             int vision_token_count,
                                             const std::vector<int>& expected_tokens,
                                             int max_tokens,
                                             float repetition_penalty,
                                             TextDecodeResult* result,
                                             std::string* error,
                                             const TextTokenCallback& token_callback) const
{
    if (!validate_text_runtime_request(ready_, result, "result pointer is null", error))
    {
        return false;
    }

    TextDecodeTiming timing;
    const auto total_start = Clock::now();
    const bool ok = decode_from_prompt_tokens(*text_embed_net_,
                                              *text_decoder_net_,
                                              *lm_head_net_,
                                              input_ids,
                                              position_ids,
                                              image_token_id,
                                              vision_features,
                                              vision_token_count,
                                              expected_tokens,
                                              eos_ids_,
                                              max_tokens,
                                              repetition_penalty,
                                              result,
                                              &timing,
                                              error,
                                              token_callback);
    if (ok)
    {
        timing.total_ms = elapsed_ms(total_start, Clock::now());
        result->timing = timing;
    }
    return ok;
}

namespace detail {

bool run_decoder_step(const ncnn::Net& net,
                      const ncnn::Mat& current_embed,
                      const ncnn::Mat& mask,
                      const ncnn::Mat& cos,
                      const ncnn::Mat& sin,
                      const DecoderKVCache& caches,
                      ncnn::Mat* hidden,
                      DecoderKVCache* updated,
                      std::array<ncnn::Mat, 4>* target_hidden,
                      std::string* error)
{
    return run_decoder_step_impl(net,
                                 current_embed,
                                 mask,
                                 cos,
                                 sin,
                                 caches,
                                 hidden,
                                 updated,
                                 target_hidden,
                                 error);
}

bool load_text_decoder_kv_net(const std::string& model_root,
                              int num_threads,
                              ncnn::Net* net,
                              std::string* error,
                              bool mmap_weights,
                              std::shared_ptr<MappedModelFile>* model_mapping,
                              bool use_vulkan,
                              int vulkan_device)
{
    if (net == nullptr)
    {
        if (error) *error = "decoder net pointer is null";
        return false;
    }
    const std::filesystem::path root = path_from_utf8(model_root);
    if (net->register_custom_layer("SDPA", create_precise_sdpa_layer) != 0)
    {
        if (error) *error = "failed to register precise SDPA layer";
        return false;
    }
    if (net->register_custom_layer("RotaryEmbed", create_multimodal_rope_layer) != 0)
    {
        if (error) *error = "failed to register multimodal RotaryEmbed layer";
        return false;
    }
    return load_net(*net,
                    root / "text_decoder" / "text_decoder_kv.ncnn.param",
                    root / "text_decoder" / "text_decoder_kv.ncnn.bin",
                    num_threads,
                    mmap_weights,
                    use_vulkan,
                    vulkan_device,
                    model_mapping,
                    error);
}

} // namespace detail

} // namespace hunyuan_ocr
