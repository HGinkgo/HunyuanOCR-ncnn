#pragma once

#include "hunyuan_ocr/dflash_runtime.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>

namespace hunyuan_ocr {

namespace detail {

std::vector<float> build_mrope(const std::vector<int>& position_ids, int seq_len, bool use_cos);
bool extract_dflash_target_hidden(ncnn::Extractor& ex,
                                  std::array<ncnn::Mat, 4>* hidden,
                                  std::string* error);

} // namespace detail

struct TextRuntimeSmokeResult {
    int token_id = -1;
    size_t embedding_values = 0;
    size_t logits_values = 0;
    int embedding_w = 0;
    int embedding_h = 0;
    int embedding_c = 0;
    int embedding_elempack = 0;
    int logits_w = 0;
    int logits_h = 0;
    int logits_c = 0;
    int logits_elempack = 0;
    int raw_top1 = -1;
    float raw_top1_score = 0.0f;
};

struct TextDecodeTiming {
    double text_embed_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double lm_head_ms = 0.0;
    double token_select_ms = 0.0;
    double total_ms = 0.0;
};

struct TextDecodeResult {
    int seq_len = 0;
    int checked_tokens = 0;
    float repetition_penalty = 1.08f;
    TextDecodeTiming timing;
    std::vector<int> generated_tokens;
    std::vector<int> raw_top1_tokens;
    std::vector<int> expected_tokens;

    bool matches_expected() const;
};

struct DFlashBlockProbeResult {
    int seq_len = 0;
    int first_token = -1;
    int acceptance_length = -1;
    int correction_token = -1;
    bool first_token_matches_expected = false;
    bool first_target_token_matches_expected = false;
    std::vector<int> proposed_tokens;
    std::vector<int> target_tokens;
};

struct DFlashDecodeTiming {
    double prefill_ms = 0.0;
    double draft_prepare_ms = 0.0;
    double draft_infer_ms = 0.0;
    double draft_postprocess_ms = 0.0;
    double verify_prepare_ms = 0.0;
    double verify_infer_ms = 0.0;
    double verify_postprocess_ms = 0.0;
    double commit_ms = 0.0;
    double total_ms = 0.0;
};

struct DFlashDecodeResult {
    TextDecodeResult decode;
    int block_count = 0;
    int drafted_token_count = 0;
    int accepted_draft_token_count = 0;
    std::vector<int> acceptance_lengths;
    DFlashDecodeTiming timing;
};

class TextRuntime {
public:
    explicit TextRuntime(int num_threads = 0);

    bool load(const std::string& model_root, std::string* error);
    bool load_dflash(const std::string& model_root, std::string* error);
    bool ready() const;
    bool dflash_ready() const;
    TextRuntimeSmokeResult smoke_token(int token_id, std::string* error) const;
    bool run_fixture_decode(const std::string& fixture_dir,
                            int max_tokens,
                            TextDecodeResult* result,
                            std::string* error) const;
    bool run_vlm_fixture_decode(const std::string& fixture_dir,
                                int max_tokens,
                                TextDecodeResult* result,
                                std::string* error) const;
    bool run_vlm_fixture_dflash_probe(const std::string& fixture_dir,
                                      DFlashBlockProbeResult* result,
                                      std::string* error) const;
    bool run_vlm_fixture_dflash_decode(const std::string& fixture_dir,
                                       int max_tokens,
                                       DFlashDecodeResult* result,
                                       std::string* error) const;
    bool run_vlm_fixture_dflash_decode_with_features(
        const std::string& fixture_dir,
        const std::vector<float>& vision_features,
        int vision_token_count,
        int max_tokens,
        DFlashDecodeResult* result,
        std::string* error) const;
    bool run_vlm_dflash_decode_with_prompt(
        const std::vector<int>& input_ids,
        const std::vector<int>& position_ids,
        int image_token_id,
        const std::vector<float>& vision_features,
        int vision_token_count,
        const std::vector<int>& expected_tokens,
        int max_tokens,
        float repetition_penalty,
        DFlashDecodeResult* result,
        std::string* error) const;
    bool run_vlm_fixture_decode_with_features(const std::string& fixture_dir,
                                              const std::vector<float>& vision_features,
                                              int vision_token_count,
                                              int max_tokens,
                                              TextDecodeResult* result,
                                              std::string* error) const;
    bool run_vlm_decode_with_prompt(const std::vector<int>& input_ids,
                                    const std::vector<int>& position_ids,
                                    int image_token_id,
                                    const std::vector<float>& vision_features,
                                    int vision_token_count,
                                    const std::vector<int>& expected_tokens,
                                    int max_tokens,
                                    float repetition_penalty,
                                    TextDecodeResult* result,
                                    std::string* error) const;

private:
    std::unique_ptr<ncnn::Net> text_embed_net_;
    std::unique_ptr<ncnn::Net> text_decoder_net_;
    std::unique_ptr<ncnn::Net> lm_head_net_;
    std::unique_ptr<DFlashDraftRuntime> dflash_draft_;
    std::vector<int> eos_ids_;
    int num_threads_ = 0;
    bool ready_ = false;
};

} // namespace hunyuan_ocr
