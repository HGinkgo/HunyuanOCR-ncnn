#include "hunyuan_ocr/dflash_runtime.h"

#include "hunyuan_ocr/hunyuan_ocr.h"
#include "hunyuan_ocr/utf8.h"
#include "mapped_model_file.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>

namespace hunyuan_ocr {
namespace detail {
namespace {

constexpr float kMaskNegInf = -1.0e38f;

} // namespace

std::vector<int> build_dflash_block(int first_token)
{
    std::vector<int> block(kDFlashBlockSize, kDFlashMaskTokenId);
    block[0] = first_token;
    return block;
}

std::vector<float> build_dflash_rope(int seq_len, bool use_cos)
{
    if (seq_len <= 0)
    {
        return {};
    }
    constexpr int head_dim = 128;
    constexpr float rope_theta = 10000.0f;
    std::vector<float> values(static_cast<size_t>(seq_len) * head_dim);
    for (int position = 0; position < seq_len; ++position)
    {
        for (int dim = 0; dim < head_dim; ++dim)
        {
            const int frequency_dim = dim % (head_dim / 2);
            const float exponent = static_cast<float>(2 * frequency_dim) / head_dim;
            const float angle = static_cast<float>(position) / std::pow(rope_theta, exponent);
            values[static_cast<size_t>(position) * head_dim + dim] =
                use_cos ? std::cos(angle) : std::sin(angle);
        }
    }
    return values;
}

std::vector<float> build_dflash_attention_mask(int context_len)
{
    if (context_len <= 0)
    {
        return {};
    }
    const int total_len = context_len + kDFlashBlockSize;
    return std::vector<float>(static_cast<size_t>(kDFlashBlockSize) * total_len, 0.0f);
}

std::vector<float> build_dflash_verify_mask(int past_len)
{
    if (past_len < 0)
    {
        return {};
    }
    const int key_len = past_len + kDFlashBlockSize;
    std::vector<float> mask(static_cast<size_t>(kDFlashBlockSize) * key_len, 0.0f);
    for (int row = 0; row < kDFlashBlockSize; ++row)
    {
        for (int col = past_len + row + 1; col < key_len; ++col)
        {
            mask[static_cast<size_t>(row) * key_len + col] = kMaskNegInf;
        }
    }
    return mask;
}

int dflash_acceptance_length(const std::vector<int>& proposed,
                             const std::vector<int>& posterior)
{
    if (proposed.empty() || proposed.size() != posterior.size())
    {
        return -1;
    }
    int accepted = 0;
    for (size_t index = 1; index < proposed.size(); ++index)
    {
        if (proposed[index] != posterior[index - 1])
        {
            break;
        }
        ++accepted;
    }
    return accepted;
}

bool view_dflash_rows(const ncnn::Mat& source,
                      int rows,
                      ncnn::Mat* output,
                      std::string* error)
{
    if (output == nullptr ||
        source.dims != 3 ||
        source.elemsize != 4u ||
        source.elempack != 1 ||
        rows <= 0 ||
        rows > source.h)
    {
        if (error) *error = "invalid DFlash row view";
        return false;
    }

    // Shallow copy keeps refcount/lifetime, data pointer, and cstep/nstep.
    // Only h is shortened so channel starts stay at source.cstep boundaries.
    ncnn::Mat view = source;
    view.h = rows;
    *output = std::move(view);
    return true;
}

bool append_dflash_rows(ncnn::Mat* target,
                        const ncnn::Mat& additions,
                        int rows,
                        std::string* error)
{
    if (target == nullptr || target->empty() ||
        (target->dims != 2 && target->dims != 3) ||
        target->dims != additions.dims ||
        target->w != additions.w || target->c != additions.c ||
        target->elemsize != 4u || additions.elemsize != 4u ||
        target->elempack != 1 || additions.elempack != 1 ||
        rows <= 0 || rows > additions.h)
    {
        if (error) *error = "invalid DFlash row append";
        return false;
    }

    const int old_rows = target->h;
    ncnn::Mat combined;
    if (target->dims == 2)
    {
        combined.create(target->w, old_rows + rows);
    }
    else
    {
        combined.create(target->w, old_rows + rows, target->c);
    }
    if (combined.empty())
    {
        if (error) *error = "DFlash row append allocation failed";
        return false;
    }

    const int channels = target->dims == 3 ? target->c : 1;
    const size_t old_bytes = static_cast<size_t>(target->w) * old_rows * sizeof(float);
    const size_t append_bytes = static_cast<size_t>(target->w) * rows * sizeof(float);
    for (int channel = 0; channel < channels; ++channel)
    {
        std::memcpy(combined.channel(channel), target->channel(channel), old_bytes);
        std::memcpy(combined.channel(channel).row(old_rows),
                    additions.channel(channel),
                    append_bytes);
    }
    *target = std::move(combined);
    return true;
}

} // namespace detail

DFlashDraftRuntime::DFlashDraftRuntime(int num_threads, bool mmap_weights)
    : net_(new ncnn::Net), num_threads_(num_threads), mmap_weights_(mmap_weights)
{
}

bool DFlashDraftRuntime::load(const std::string& param_path,
                              const std::string& bin_path,
                              std::string* error)
{
    ready_ = false;
    std::unique_ptr<ncnn::Net> candidate(new ncnn::Net);
    candidate->opt = make_fp32_ncnn_option(num_threads_);
    candidate->opt.use_packing_layout = false;

    const std::filesystem::path param = path_from_utf8(param_path);
    const std::filesystem::path bin = path_from_utf8(bin_path);
    if (candidate->load_param(param.c_str()) != 0)
    {
        if (error) *error = "failed to load DFlash param: " + param_path;
        return false;
    }
    std::shared_ptr<MappedModelFile> candidate_mapping;
    if (!load_model_file(*candidate, bin, mmap_weights_, &candidate_mapping, error))
    {
        return false;
    }

    net_ = std::move(candidate);
    model_mapping_ = std::move(candidate_mapping);
    ready_ = true;
    return true;
}

bool DFlashDraftRuntime::ready() const
{
    return ready_;
}

size_t DFlashDraftRuntime::mapped_weight_bytes() const
{
    return model_mapping_ ? model_mapping_->size() : 0;
}

bool DFlashDraftRuntime::run(const DFlashDraftInput& input,
                             ncnn::Mat* output,
                             std::string* error) const
{
    if (!ready_)
    {
        if (error) *error = "DFlash draft runtime is not loaded";
        return false;
    }
    if (output == nullptr)
    {
        if (error) *error = "DFlash output pointer is null";
        return false;
    }

    const int context_len = input.target_hidden[0].h;
    const int total_len = context_len + detail::kDFlashBlockSize;
    if (input.noise_embedding.w != 1024 ||
        input.noise_embedding.h != detail::kDFlashBlockSize ||
        context_len <= 0)
    {
        if (error) *error = "invalid DFlash noise/context shape";
        return false;
    }
    for (const ncnn::Mat& hidden : input.target_hidden)
    {
        if (hidden.w != 1024 || hidden.h != context_len)
        {
            if (error) *error = "DFlash target hidden shapes do not match";
            return false;
        }
    }
    if (input.cos.dims != 3 || input.cos.c != 1 ||
        input.cos.w != 128 || input.cos.h != total_len ||
        input.sin.dims != 3 || input.sin.c != 1 ||
        input.sin.w != 128 || input.sin.h != total_len ||
        input.attention_mask.dims != 3 || input.attention_mask.c != 1 ||
        input.attention_mask.w != total_len ||
        input.attention_mask.h != detail::kDFlashBlockSize)
    {
        if (error) *error = "invalid DFlash rope/mask shape";
        return false;
    }

    ncnn::Extractor ex = net_->create_extractor();
    if (ex.input("in0", input.noise_embedding) != 0)
    {
        if (error) *error = "DFlash noise input failed";
        return false;
    }
    for (size_t index = 0; index < input.target_hidden.size(); ++index)
    {
        const std::string name = "in" + std::to_string(index + 1);
        if (ex.input(name.c_str(), input.target_hidden[index]) != 0)
        {
            if (error) *error = "DFlash target hidden input failed: " + name;
            return false;
        }
    }
    if (ex.input("in5", input.cos) != 0 ||
        ex.input("in6", input.sin) != 0 ||
        ex.input("in7", input.attention_mask) != 0)
    {
        if (error) *error = "DFlash rope/mask input failed";
        return false;
    }
    if (ex.extract("out0", *output) != 0)
    {
        if (error) *error = "DFlash output extraction failed";
        return false;
    }
    if (output->w != 1024 || output->h != detail::kDFlashBlockSize)
    {
        if (error) *error = "DFlash output shape mismatch";
        return false;
    }
    return true;
}

} // namespace hunyuan_ocr
