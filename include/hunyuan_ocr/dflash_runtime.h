#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>

namespace hunyuan_ocr {

namespace detail {

constexpr int kDFlashBlockSize = 16;
constexpr int kDFlashMaskTokenId = 120817;

std::vector<int> build_dflash_block(int first_token);
std::vector<float> build_dflash_rope(int seq_len, bool use_cos);
std::vector<float> build_dflash_attention_mask(int context_len);
std::vector<float> build_dflash_verify_mask(int past_len);
int dflash_acceptance_length(const std::vector<int>& proposed,
                             const std::vector<int>& posterior);
// Zero-copy prefix view over a 3D fp32 pack1 cache: shares storage and keeps
// source cstep/nstep so channel strides remain valid when only h is shortened.
bool view_dflash_rows(const ncnn::Mat& source,
                      int rows,
                      ncnn::Mat* output,
                      std::string* error);
bool append_dflash_rows(ncnn::Mat* target,
                        const ncnn::Mat& additions,
                        int rows,
                        std::string* error);

} // namespace detail

struct DFlashDraftInput {
    ncnn::Mat noise_embedding;
    std::array<ncnn::Mat, 4> target_hidden;
    ncnn::Mat cos;
    ncnn::Mat sin;
    ncnn::Mat attention_mask;
};

class DFlashDraftRuntime {
public:
    explicit DFlashDraftRuntime(int num_threads = 0);

    bool load(const std::string& param_path,
              const std::string& bin_path,
              std::string* error);
    bool ready() const;
    bool run(const DFlashDraftInput& input,
             ncnn::Mat* output,
             std::string* error) const;

private:
    std::unique_ptr<ncnn::Net> net_;
    int num_threads_ = 0;
    bool ready_ = false;
};

} // namespace hunyuan_ocr
