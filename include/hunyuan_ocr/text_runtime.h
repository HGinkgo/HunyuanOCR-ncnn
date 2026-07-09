#pragma once

#include <memory>
#include <string>
#include <vector>

#include <mat.h>
#include <net.h>

namespace hunyuan_ocr {

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
    double total_ms = 0.0;
};

struct TextDecodeResult {
    int seq_len = 0;
    int checked_tokens = 0;
    float repetition_penalty = 1.03f;
    TextDecodeTiming timing;
    std::vector<int> generated_tokens;
    std::vector<int> raw_top1_tokens;
    std::vector<int> expected_tokens;

    bool matches_expected() const;
};

class TextRuntime {
public:
    TextRuntime();

    bool load(const std::string& model_root, std::string* error);
    bool ready() const;
    TextRuntimeSmokeResult smoke_token(int token_id, std::string* error) const;
    bool run_fixture_decode(const std::string& fixture_dir,
                            int max_tokens,
                            TextDecodeResult* result,
                            std::string* error) const;
    bool run_vlm_fixture_decode(const std::string& fixture_dir,
                                int max_tokens,
                                TextDecodeResult* result,
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
                                    TextDecodeResult* result,
                                    std::string* error) const;

private:
    std::unique_ptr<ncnn::Net> text_embed_net_;
    std::unique_ptr<ncnn::Net> text_decoder_net_;
    std::unique_ptr<ncnn::Net> lm_head_net_;
    bool ready_ = false;
};

} // namespace hunyuan_ocr
