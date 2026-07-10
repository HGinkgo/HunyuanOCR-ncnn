#include "hunyuan_ocr/text_runtime.h"

#include "hunyuan_ocr/hunyuan_ocr.h"

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

namespace hunyuan_ocr {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

constexpr int kHiddenSize = 1024;
constexpr int kHeadDim = 128;
constexpr int kAttentionLayerCount = 24;
constexpr int kVocabSize = 120818;
constexpr float kRopeTheta = 10000.0f;
constexpr float kRopeAlpha = 1000.0f;
constexpr float kMaskNegInf = -1.0e38f;
const int kXdRopeSection[4] = {16, 16, 16, 16};
const int kEosIds[2] = {120007, 120020};

bool load_net(ncnn::Net& net,
              const std::filesystem::path& param,
              const std::filesystem::path& bin,
              int num_threads,
              std::string* error)
{
    net.opt = make_fp32_ncnn_option(num_threads);
    net.opt.use_packing_layout = false;

    if (net.load_param(param.string().c_str()) != 0)
    {
        if (error) *error = "failed to load param: " + param.string();
        return false;
    }
    if (net.load_model(bin.string().c_str()) != 0)
    {
        if (error) *error = "failed to load bin: " + bin.string();
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

float bf16_round_rne(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    const uint32_t rounding_bias = ((bits >> 16) & 1u) + 0x7FFFu;
    bits = (bits + rounding_bias) & 0xFFFF0000u;
    float rounded = 0.0f;
    std::memcpy(&rounded, &bits, sizeof(float));
    return rounded;
}

float rope_value(int position, int dim, bool use_cos)
{
    const double base = static_cast<double>(kRopeTheta) *
                        std::pow(static_cast<double>(kRopeAlpha),
                                 static_cast<double>(kHeadDim) / static_cast<double>(kHeadDim - 2));
    const int half_dim = dim % (kHeadDim / 2);
    const float exponent = static_cast<float>(2 * half_dim) / static_cast<float>(kHeadDim);
    const float inv_freq = static_cast<float>(1.0 / std::pow(base, static_cast<double>(exponent)));
    const float angle = static_cast<float>(position) * inv_freq;
    const float value = use_cos ? static_cast<float>(std::cos(angle)) : static_cast<float>(std::sin(angle));
    return bf16_round_rne(value);
}

ncnn::Mat make_mat_from_vector(int w, int h, const std::vector<float>& values)
{
    ncnn::Mat mat(w, h, const_cast<float*>(values.data()));
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

std::vector<float> build_1d_rope(int seq_len, bool use_cos)
{
    std::vector<float> values(static_cast<size_t>(seq_len) * kHeadDim);
    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int dim = 0; dim < kHeadDim; ++dim)
        {
            values[static_cast<size_t>(pos) * kHeadDim + dim] = rope_value(pos, dim, use_cos);
        }
    }
    return values;
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

std::vector<float> build_xd_rope(const std::vector<int>& position_ids, int seq_len, bool use_cos)
{
    std::vector<float> values(static_cast<size_t>(seq_len) * kHeadDim);
    int out_dim = 0;
    for (int split = 0; split < 8; ++split)
    {
        const int axis = split % 4;
        const int section = kXdRopeSection[split % 4];
        for (int local = 0; local < section; ++local)
        {
            const int source_dim = out_dim + local;
            for (int pos_index = 0; pos_index < seq_len; ++pos_index)
            {
                const int position = position_ids[static_cast<size_t>(axis) * seq_len + pos_index];
                values[static_cast<size_t>(pos_index) * kHeadDim + out_dim + local] =
                    rope_value(position, source_dim, use_cos);
            }
        }
        out_dim += section;
    }
    return values;
}

template <typename T>
bool read_binary_vector(const std::filesystem::path& path, size_t expected_count, std::vector<T>* values, std::string* error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        if (error) *error = "failed to open: " + path.string();
        return false;
    }
    values->assign(expected_count, T{});
    file.read(reinterpret_cast<char*>(values->data()), static_cast<std::streamsize>(expected_count * sizeof(T)));
    if (file.gcount() != static_cast<std::streamsize>(expected_count * sizeof(T)))
    {
        if (error) *error = "short read: " + path.string();
        return false;
    }
    char extra = 0;
    if (file.read(&extra, 1))
    {
        if (error) *error = "unexpected extra bytes: " + path.string();
        return false;
    }
    return true;
}

bool parse_fixture_meta(const std::filesystem::path& path, int* seq_len, int* expected_token_count, std::string* error)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        if (error) *error = "failed to open fixture meta: " + path.string();
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
            if (key == "seq_len") *seq_len = std::stoi(value);
            else if (key == "expected_token_count") *expected_token_count = std::stoi(value);
        }
        catch (const std::exception&)
        {
            if (error) *error = "invalid fixture meta value: " + line;
            return false;
        }
    }

    if (*seq_len <= 0 || *expected_token_count <= 0)
    {
        if (error) *error = "fixture meta must define positive seq_len and expected_token_count";
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
        if (error) *error = "failed to open fixture meta: " + path.string();
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
    ncnn::Mat input_ids_mat(seq_len, 1, const_cast<int*>(input_ids.data()));
    input_ids_mat = input_ids_mat.clone();

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

using KVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;

bool run_decoder_prefill(const ncnn::Net& net,
                         const ncnn::Mat& inputs_embeds,
                         const ncnn::Mat& mask,
                         const ncnn::Mat& xd_cos,
                         const ncnn::Mat& xd_sin,
                         const ncnn::Mat& rope_cos,
                         const ncnn::Mat& rope_sin,
                         ncnn::Mat* hidden,
                         KVCache* caches,
                         std::string* error)
{
    ncnn::Extractor ex = const_cast<ncnn::Net&>(net).create_extractor();
    if (ex.input("in0", inputs_embeds) != 0 ||
        ex.input("in1", mask) != 0 ||
        ex.input("in2", xd_cos) != 0 ||
        ex.input("in3", xd_sin) != 0 ||
        ex.input("in4", rope_cos) != 0 ||
        ex.input("in5", rope_sin) != 0)
    {
        if (error) *error = "decoder prefill input failed";
        return false;
    }

    caches->clear();
    caches->reserve(kAttentionLayerCount);
    for (int i = 0; i < kAttentionLayerCount; ++i)
    {
        char name_k[32];
        char name_v[32];
        std::snprintf(name_k, sizeof(name_k), "out_cache_k%d", i);
        std::snprintf(name_v, sizeof(name_v), "out_cache_v%d", i);
        ncnn::Mat key;
        ncnn::Mat value;
        if (!extract_or_error(ex, name_k, &key, error) ||
            !extract_or_error(ex, name_v, &value, error))
        {
            return false;
        }
        caches->emplace_back(std::move(key), std::move(value));
    }
    return extract_or_error(ex, "out0", hidden, error);
}

bool run_decoder_step(const ncnn::Net& net,
                      const ncnn::Mat& current_embed,
                      const ncnn::Mat& mask,
                      const ncnn::Mat& cos,
                      const ncnn::Mat& sin,
                      const KVCache& caches,
                      ncnn::Mat* hidden,
                      KVCache* updated,
                      std::string* error)
{
    ncnn::Extractor ex = const_cast<ncnn::Net&>(net).create_extractor();
    if (ex.input("in0", current_embed) != 0 ||
        ex.input("in1", mask) != 0 ||
        ex.input("in2", cos) != 0 ||
        ex.input("in3", sin) != 0 ||
        ex.input("in4", cos) != 0 ||
        ex.input("in5", sin) != 0)
    {
        if (error) *error = "decoder step input failed";
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
            if (error) *error = std::string("decoder step cache input failed at layer ") + std::to_string(i);
            return false;
        }
    }

    updated->clear();
    updated->reserve(kAttentionLayerCount);
    for (int i = 0; i < kAttentionLayerCount; ++i)
    {
        char name_k[32];
        char name_v[32];
        std::snprintf(name_k, sizeof(name_k), "out_cache_k%d", i);
        std::snprintf(name_v, sizeof(name_v), "out_cache_v%d", i);
        ncnn::Mat key;
        ncnn::Mat value;
        if (!extract_or_error(ex, name_k, &key, error) ||
            !extract_or_error(ex, name_v, &value, error))
        {
            return false;
        }
        updated->emplace_back(std::move(key), std::move(value));
    }
    return extract_or_error(ex, "out0", hidden, error);
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

bool is_eos(int token)
{
    for (const int eos : kEosIds)
    {
        if (token == eos) return true;
    }
    return false;
}

bool decode_from_embeddings(const ncnn::Net& text_embed_net,
                            const ncnn::Net& decoder_net,
                            const ncnn::Net& lm_head_net,
                            int seq_len,
                            int max_tokens,
                            const std::vector<float>& inputs_embeds,
                            const std::vector<int>& input_ids,
                            const std::vector<int>& position_ids,
                            const std::vector<int>& expected_tokens,
                            TextDecodeResult* result,
                            TextDecodeTiming* timing,
                            std::string* error)
{
    const auto total_start = Clock::now();
    if (seq_len <= 0)
    {
        if (error) *error = "seq_len must be positive";
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
    const std::vector<float> xd_cos_data = build_xd_rope(position_ids, seq_len, true);
    const std::vector<float> xd_sin_data = build_xd_rope(position_ids, seq_len, false);
    const std::vector<float> rope_cos_data = build_1d_rope(seq_len, true);
    const std::vector<float> rope_sin_data = build_1d_rope(seq_len, false);
    const ncnn::Mat xd_cos = make_mat_from_vector(kHeadDim, seq_len, xd_cos_data);
    const ncnn::Mat xd_sin = make_mat_from_vector(kHeadDim, seq_len, xd_sin_data);
    const ncnn::Mat rope_cos = make_mat_from_vector(kHeadDim, seq_len, rope_cos_data);
    const ncnn::Mat rope_sin = make_mat_from_vector(kHeadDim, seq_len, rope_sin_data);

    ncnn::Mat prefill_hidden;
    KVCache caches;
    const auto prefill_start = Clock::now();
    if (!run_decoder_prefill(decoder_net, input_mat, prefill_mask, xd_cos, xd_sin, rope_cos, rope_sin,
                             &prefill_hidden, &caches, error))
    {
        return false;
    }
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
    local.repetition_penalty = 1.03f;
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

        if (is_eos(current_token) || out_index + 1 >= max_tokens)
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
        KVCache updated;
        if (!run_decoder_step(decoder_net, current_embed, decode_mask, decode_cos, decode_sin,
                              caches, &decode_hidden, &updated, error))
        {
            return false;
        }
        caches = std::move(updated);
        if (timing)
        {
            timing->decode_ms += elapsed_ms(decoder_step_start, Clock::now());
        }

        const auto lm_head_start = Clock::now();
        if (!run_lm_head(lm_head_net, decode_hidden, &logits, error))
        {
            return false;
        }
        if (timing)
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
                              int max_tokens,
                              TextDecodeResult* result,
                              TextDecodeTiming* timing,
                              std::string* error)
{
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

    std::vector<float> inputs_embeds;
    const auto text_embed_start = Clock::now();
    if (!run_text_embed_tokens(text_embed_net, input_ids, seq_len, &inputs_embeds, error))
    {
        return false;
    }
    int injected = 0;
    for (int i = 0; i < seq_len; ++i)
    {
        if (input_ids[static_cast<size_t>(i)] != image_token_id)
        {
            continue;
        }
        if (injected >= vision_token_count)
        {
            if (error) *error = "more image tokens than vision features";
            return false;
        }
        std::memcpy(inputs_embeds.data() + static_cast<size_t>(i) * kHiddenSize,
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
    if (timing)
    {
        timing->text_embed_ms += elapsed_ms(text_embed_start, Clock::now());
    }

    return decode_from_embeddings(text_embed_net,
                                  decoder_net,
                                  lm_head_net,
                                  seq_len,
                                  max_tokens,
                                  inputs_embeds,
                                  input_ids,
                                  position_ids,
                                  expected_tokens,
                                  result,
                                  timing,
                                  error);
}

} // namespace

TextRuntime::TextRuntime(int num_threads)
    : text_embed_net_(new ncnn::Net),
      text_decoder_net_(new ncnn::Net),
      lm_head_net_(new ncnn::Net),
      num_threads_(num_threads)
{
}

bool TextRuntime::load(const std::string& model_root, std::string* error)
{
    const std::filesystem::path root(model_root);
    ready_ = false;

    if (!load_net(*text_embed_net_,
                  root / "text_embed" / "text_embed.ncnn.param",
                  root / "text_embed" / "text_embed.ncnn.bin",
                  num_threads_,
                  error))
    {
        return false;
    }
    if (!load_net(*text_decoder_net_,
                  root / "text_decoder" / "text_decoder_kv.ncnn.param",
                  root / "text_decoder" / "text_decoder_kv.ncnn.bin",
                  num_threads_,
                  error))
    {
        return false;
    }
    if (!load_net(*lm_head_net_,
                  root / "lm_head" / "lm_head.ncnn.param",
                  root / "lm_head" / "lm_head.ncnn.bin",
                  num_threads_,
                  error))
    {
        return false;
    }

    ready_ = true;
    return true;
}

bool TextRuntime::ready() const
{
    return ready_;
}

bool TextDecodeResult::matches_expected() const
{
    return generated_tokens == expected_tokens;
}

TextRuntimeSmokeResult TextRuntime::smoke_token(int token_id, std::string* error) const
{
    TextRuntimeSmokeResult result;
    result.token_id = token_id;

    if (!ready_)
    {
        if (error) *error = "text runtime is not loaded";
        return result;
    }

    ncnn::Mat token_mat(1, 1, 1);
    *static_cast<int*>(token_mat.data) = token_id;

    ncnn::Mat embedding;
    {
        ncnn::Extractor ex = text_embed_net_->create_extractor();
        if (ex.input("in0", token_mat) != 0)
        {
            if (error) *error = "text_embed input failed";
            return result;
        }
        if (ex.extract("out0", embedding) != 0)
        {
            if (error) *error = "text_embed extract failed";
            return result;
        }
    }
    result.embedding_values = logical_value_count(embedding);
    result.embedding_w = embedding.w;
    result.embedding_h = embedding.h;
    result.embedding_c = embedding.c;
    result.embedding_elempack = embedding.elempack;

    ncnn::Mat logits;
    {
        ncnn::Extractor ex = lm_head_net_->create_extractor();
        if (ex.input("in0", embedding) != 0)
        {
            if (error) *error = "lm_head input failed";
            return result;
        }
        if (ex.extract("out0", logits) != 0)
        {
            if (error) *error = "lm_head extract failed";
            return result;
        }
    }
    result.logits_values = logical_value_count(logits);
    result.logits_w = logits.w;
    result.logits_h = logits.h;
    result.logits_c = logits.c;
    result.logits_elempack = logits.elempack;

    const float* scores = logits;
    if (scores != nullptr && result.logits_values > 0)
    {
        const size_t count = result.logits_values;
        size_t best = 0;
        for (size_t i = 1; i < count; ++i)
        {
            if (scores[i] > scores[best])
            {
                best = i;
            }
        }
        result.raw_top1 = static_cast<int>(best);
        result.raw_top1_score = scores[best];
    }

    return result;
}

bool TextRuntime::run_fixture_decode(const std::string& fixture_dir,
                                     int max_tokens,
                                     TextDecodeResult* result,
                                     std::string* error) const
{
    if (!ready_)
    {
        if (error) *error = "text runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }

    const std::filesystem::path root(fixture_dir);
    int seq_len = 0;
    int expected_token_count = 0;
    if (!parse_fixture_meta(root / "meta.txt", &seq_len, &expected_token_count, error))
    {
        return false;
    }
    if (max_tokens <= 0 || max_tokens > expected_token_count)
    {
        max_tokens = expected_token_count;
    }

    std::vector<float> inputs_embeds;
    std::vector<int> input_ids;
    std::vector<int> position_ids;
    std::vector<int> expected_tokens;
    if (!read_binary_vector(root / "inputs_embeds.f32", static_cast<size_t>(seq_len) * kHiddenSize, &inputs_embeds, error) ||
        !read_binary_vector(root / "input_ids.i32", static_cast<size_t>(seq_len), &input_ids, error) ||
        !read_binary_vector(root / "position_ids.i32", static_cast<size_t>(seq_len) * 4, &position_ids, error) ||
        !read_binary_vector(root / "expected_tokens.i32", static_cast<size_t>(expected_token_count), &expected_tokens, error))
    {
        return false;
    }

    return decode_from_embeddings(*text_embed_net_,
                                  *text_decoder_net_,
                                  *lm_head_net_,
                                  seq_len,
                                  max_tokens,
                                  inputs_embeds,
                                  input_ids,
                                  position_ids,
                                  expected_tokens,
                                  result,
                                  nullptr,
                                  error);
}

bool TextRuntime::run_vlm_fixture_decode(const std::string& fixture_dir,
                                         int max_tokens,
                                         TextDecodeResult* result,
                                         std::string* error) const
{
    if (!ready_)
    {
        if (error) *error = "text runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }

    const std::filesystem::path root(fixture_dir);
    VlmFixtureMeta meta;
    if (!parse_vlm_fixture_meta(root / "meta.txt", &meta, error))
    {
        return false;
    }

    std::vector<int> input_ids;
    std::vector<int> position_ids;
    std::vector<int> expected_tokens;
    std::vector<float> vision_features;
    if (!read_binary_vector(root / "input_ids.i32", static_cast<size_t>(meta.seq_len), &input_ids, error) ||
        !read_binary_vector(root / "position_ids.i32", static_cast<size_t>(meta.seq_len) * 4, &position_ids, error) ||
        !read_binary_vector(root / "expected_tokens.i32", static_cast<size_t>(meta.expected_token_count), &expected_tokens, error) ||
        !read_binary_vector(root / "vision_features.f32", static_cast<size_t>(meta.vision_token_count) * kHiddenSize, &vision_features, error))
    {
        return false;
    }

    std::vector<float> inputs_embeds;
    if (!run_text_embed_tokens(*text_embed_net_, input_ids, meta.seq_len, &inputs_embeds, error))
    {
        return false;
    }

    int injected = 0;
    for (int i = 0; i < meta.seq_len; ++i)
    {
        if (input_ids[static_cast<size_t>(i)] != meta.image_token_id)
        {
            continue;
        }
        if (injected >= meta.vision_token_count)
        {
            if (error) *error = "more image tokens than vision features";
            return false;
        }
        std::memcpy(inputs_embeds.data() + static_cast<size_t>(i) * kHiddenSize,
                    vision_features.data() + static_cast<size_t>(injected) * kHiddenSize,
                    static_cast<size_t>(kHiddenSize) * sizeof(float));
        ++injected;
    }
    if (injected != meta.vision_token_count)
    {
        if (error) {
            *error = "image token count mismatch: injected " + std::to_string(injected) +
                     ", expected " + std::to_string(meta.vision_token_count);
        }
        return false;
    }

    return decode_from_embeddings(*text_embed_net_,
                                  *text_decoder_net_,
                                  *lm_head_net_,
                                  meta.seq_len,
                                  max_tokens,
                                  inputs_embeds,
                                  input_ids,
                                  position_ids,
                                  expected_tokens,
                                  result,
                                  nullptr,
                                  error);
}

bool TextRuntime::run_vlm_fixture_decode_with_features(const std::string& fixture_dir,
                                                       const std::vector<float>& vision_features,
                                                       int vision_token_count,
                                                       int max_tokens,
                                                       TextDecodeResult* result,
                                                       std::string* error) const
{
    if (!ready_)
    {
        if (error) *error = "text runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
        return false;
    }

    const std::filesystem::path root(fixture_dir);
    VlmFixtureMeta meta;
    if (!parse_vlm_fixture_meta(root / "meta.txt", &meta, error))
    {
        return false;
    }
    if (vision_token_count != meta.vision_token_count)
    {
        if (error) {
            *error = "external vision token count mismatch: got " + std::to_string(vision_token_count) +
                     ", expected " + std::to_string(meta.vision_token_count);
        }
        return false;
    }
    if (vision_features.size() != static_cast<size_t>(meta.vision_token_count) * kHiddenSize)
    {
        if (error) {
            *error = "external vision feature size mismatch: got " + std::to_string(vision_features.size()) +
                     ", expected " + std::to_string(static_cast<size_t>(meta.vision_token_count) * kHiddenSize);
        }
        return false;
    }

    std::vector<int> input_ids;
    std::vector<int> position_ids;
    std::vector<int> expected_tokens;
    if (!read_binary_vector(root / "input_ids.i32", static_cast<size_t>(meta.seq_len), &input_ids, error) ||
        !read_binary_vector(root / "position_ids.i32", static_cast<size_t>(meta.seq_len) * 4, &position_ids, error) ||
        !read_binary_vector(root / "expected_tokens.i32", static_cast<size_t>(meta.expected_token_count), &expected_tokens, error))
    {
        return false;
    }

    std::vector<float> inputs_embeds;
    if (!run_text_embed_tokens(*text_embed_net_, input_ids, meta.seq_len, &inputs_embeds, error))
    {
        return false;
    }

    int injected = 0;
    for (int i = 0; i < meta.seq_len; ++i)
    {
        if (input_ids[static_cast<size_t>(i)] != meta.image_token_id)
        {
            continue;
        }
        if (injected >= meta.vision_token_count)
        {
            if (error) *error = "more image tokens than external vision features";
            return false;
        }
        std::memcpy(inputs_embeds.data() + static_cast<size_t>(i) * kHiddenSize,
                    vision_features.data() + static_cast<size_t>(injected) * kHiddenSize,
                    static_cast<size_t>(kHiddenSize) * sizeof(float));
        ++injected;
    }
    if (injected != meta.vision_token_count)
    {
        if (error) {
            *error = "image token count mismatch: injected " + std::to_string(injected) +
                     ", expected " + std::to_string(meta.vision_token_count);
        }
        return false;
    }

    return decode_from_embeddings(*text_embed_net_,
                                  *text_decoder_net_,
                                  *lm_head_net_,
                                  meta.seq_len,
                                  max_tokens,
                                  inputs_embeds,
                                  input_ids,
                                  position_ids,
                                  expected_tokens,
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
                                             TextDecodeResult* result,
                                             std::string* error) const
{
    if (!ready_)
    {
        if (error) *error = "text runtime is not loaded";
        return false;
    }
    if (result == nullptr)
    {
        if (error) *error = "result pointer is null";
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
                                              max_tokens,
                                              result,
                                              &timing,
                                              error);
    if (ok)
    {
        timing.total_ms = elapsed_ms(total_start, Clock::now());
        result->timing = timing;
    }
    return ok;
}

} // namespace hunyuan_ocr
