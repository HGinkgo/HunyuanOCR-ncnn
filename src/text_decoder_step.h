#pragma once

// Internal header for production decoder-step helpers shared with
// tests/decoder microbenchmarks. Not part of the public HunyuanOCR-ncnn API.

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mat.h>
#include <net.h>

namespace hunyuan_ocr {
class MappedModelFile;
namespace detail {

constexpr int kTextHiddenSize = 1024;
constexpr int kTextHeadDim = 128;
constexpr int kTextAttentionLayerCount = 24;
// Decoder KV tensors after the graph's GQA expand are stored with 16 heads
// (matching out_cache_k/v from production prefill/step), not the raw 8 KV heads.
constexpr int kTextCacheHeads = 16;

using DecoderKVCache = std::vector<std::pair<ncnn::Mat, ncnn::Mat>>;

// Production decoder step used by AR decode and DFlash verify.
bool run_decoder_step(const ncnn::Net& net,
                      const ncnn::Mat& current_embed,
                      const ncnn::Mat& mask,
                      const ncnn::Mat& cos,
                      const ncnn::Mat& sin,
                      const DecoderKVCache& caches,
                      ncnn::Mat* hidden,
                      DecoderKVCache* updated,
                      std::array<ncnn::Mat, 4>* target_hidden,
                      std::string* error);

#if NCNN_VULKAN
class VulkanDecoderSession {
public:
    explicit VulkanDecoderSession(const ncnn::Net& net);
    ~VulkanDecoderSession();

    VulkanDecoderSession(const VulkanDecoderSession&) = delete;
    VulkanDecoderSession& operator=(const VulkanDecoderSession&) = delete;

    bool initialize(const DecoderKVCache& caches, std::string* error);
    bool run_step(const ncnn::Mat& current_embed,
                  const ncnn::Mat& mask,
                  const ncnn::Mat& cos,
                  const ncnn::Mat& sin,
                  ncnn::Mat* hidden,
                  std::string* error);
    int step_submit_count() const;
    int extractor_create_count() const;
    int hidden_extract_count() const;
    int kv_extract_count() const;
    int command_reset_count() const;
    int input_upload_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

bool load_text_decoder_kv_net(const std::string& model_root,
                              int num_threads,
                              ncnn::Net* net,
                              std::string* error,
                              bool mmap_weights = false,
                              std::shared_ptr<MappedModelFile>* model_mapping = nullptr,
                              bool use_vulkan = false,
                              int vulkan_device = 0);

} // namespace detail
} // namespace hunyuan_ocr
