#include "text_decoder_step.h"

#if NCNN_VULKAN

#include <allocator.h>
#include <command.h>
#include <gpu.h>

#include <cstdio>
#include <utility>
#include <vector>

namespace hunyuan_ocr {
namespace detail {
namespace {

using VulkanKVCache = std::vector<std::pair<ncnn::VkMat, ncnn::VkMat>>;

ncnn::Option make_vulkan_option(const ncnn::Net& net,
                                ncnn::VkAllocator* blob_allocator,
                                ncnn::VkAllocator* staging_allocator)
{
    ncnn::Option option = net.opt;
    option.blob_vkallocator = blob_allocator;
    option.workspace_vkallocator = blob_allocator;
    option.staging_vkallocator = staging_allocator;
    return option;
}

bool bind_gpu_inputs(ncnn::Extractor* ex,
                     const ncnn::VkMat& input,
                     const ncnn::VkMat& mask,
                     const ncnn::VkMat& cos,
                     const ncnn::VkMat& sin,
                     std::string* error)
{
    if (ex == nullptr ||
        ex->input("in0", input) != 0 ||
        ex->input("in1", mask) != 0 ||
        ex->input("in2", cos) != 0 ||
        ex->input("in3", sin) != 0 ||
        ex->input("in4", cos) != 0 ||
        ex->input("in5", sin) != 0)
    {
        if (error) *error = "Vulkan decoder step input failed";
        return false;
    }
    return true;
}

} // namespace

struct VulkanDecoderSession::Impl {
    explicit Impl(const ncnn::Net& decoder_net)
        : net(decoder_net), vkdev(decoder_net.vulkan_device())
    {
        if (net.opt.use_vulkan_compute && vkdev != nullptr)
        {
            blob_allocator = vkdev->acquire_blob_allocator();
            staging_allocator = vkdev->acquire_staging_allocator();
        }
    }

    ~Impl()
    {
        command.reset();
        caches.clear();
        if (blob_allocator != nullptr)
        {
            blob_allocator->clear();
            vkdev->reclaim_blob_allocator(blob_allocator);
        }
        if (staging_allocator != nullptr)
        {
            staging_allocator->clear();
            vkdev->reclaim_staging_allocator(staging_allocator);
        }
    }

    const ncnn::Net& net;
    const ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_allocator = nullptr;
    ncnn::VkAllocator* staging_allocator = nullptr;
    std::unique_ptr<ncnn::VkCompute> command;
    VulkanKVCache caches;
    int step_submits = 0;
    int command_resets = 0;
    int input_uploads = 0;
};

VulkanDecoderSession::VulkanDecoderSession(const ncnn::Net& net)
    : impl_(new Impl(net))
{
}

VulkanDecoderSession::~VulkanDecoderSession() = default;

bool VulkanDecoderSession::initialize(const DecoderKVCache& caches, std::string* error)
{
    if (impl_->vkdev == nullptr || impl_->blob_allocator == nullptr ||
        impl_->staging_allocator == nullptr)
    {
        if (error) *error = "Vulkan decoder session is unavailable";
        return false;
    }
    if (static_cast<int>(caches.size()) != kTextAttentionLayerCount)
    {
        if (error) *error = "invalid Vulkan KV cache layer count";
        return false;
    }

    const ncnn::Option option = make_vulkan_option(
        impl_->net, impl_->blob_allocator, impl_->staging_allocator);
    ncnn::VkCompute cmd(impl_->vkdev);
    VulkanKVCache uploaded;
    uploaded.reserve(caches.size());
    for (const auto& cache : caches)
    {
        ncnn::VkMat key;
        ncnn::VkMat value;
        cmd.record_upload(cache.first, key, option);
        cmd.record_upload(cache.second, value, option);
        uploaded.emplace_back(std::move(key), std::move(value));
    }
    if (cmd.submit_and_wait() != 0)
    {
        if (error) *error = "Vulkan KV cache upload failed";
        return false;
    }
    impl_->caches = std::move(uploaded);
    return true;
}

bool VulkanDecoderSession::run_step(const ncnn::Mat& current_embed,
                                    const ncnn::Mat& mask,
                                    const ncnn::Mat& cos,
                                    const ncnn::Mat& sin,
                                    ncnn::Mat* hidden,
                                    std::string* error)
{
    if (hidden == nullptr ||
        static_cast<int>(impl_->caches.size()) != kTextAttentionLayerCount)
    {
        if (error) *error = "Vulkan decoder session is not initialized";
        return false;
    }

    const ncnn::Option option = make_vulkan_option(
        impl_->net, impl_->blob_allocator, impl_->staging_allocator);
    if (!impl_->command)
    {
        impl_->command = std::make_unique<ncnn::VkCompute>(impl_->vkdev);
    }
    else if (impl_->command->reset() != 0)
    {
        if (error) *error = "Vulkan decoder command reset failed";
        return false;
    }
    else
    {
        ++impl_->command_resets;
    }
    ncnn::VkCompute& cmd = *impl_->command;

    ncnn::Extractor ex = const_cast<ncnn::Net&>(impl_->net).create_extractor();
    ex.set_blob_vkallocator(impl_->blob_allocator);
    ex.set_workspace_vkallocator(impl_->blob_allocator);
    ex.set_staging_vkallocator(impl_->staging_allocator);
    ncnn::VkMat input_gpu;
    ncnn::VkMat mask_gpu;
    ncnn::VkMat cos_gpu;
    ncnn::VkMat sin_gpu;
    cmd.record_upload(current_embed, input_gpu, option);
    cmd.record_upload(mask, mask_gpu, option);
    cmd.record_upload(cos, cos_gpu, option);
    cmd.record_upload(sin, sin_gpu, option);
    impl_->input_uploads += 4;
    if (!bind_gpu_inputs(&ex, input_gpu, mask_gpu, cos_gpu, sin_gpu, error))
    {
        return false;
    }
    for (int layer = 0; layer < kTextAttentionLayerCount; ++layer)
    {
        char name_k[32];
        char name_v[32];
        std::snprintf(name_k, sizeof(name_k), "cache_k%d", layer);
        std::snprintf(name_v, sizeof(name_v), "cache_v%d", layer);
        const auto& cache = impl_->caches[static_cast<size_t>(layer)];
        if (ex.input(name_k, cache.first) != 0 || ex.input(name_v, cache.second) != 0)
        {
            if (error)
            {
                *error = "Vulkan decoder cache input failed at layer " +
                         std::to_string(layer);
            }
            return false;
        }
    }

    ncnn::VkMat hidden_gpu;
    if (ex.extract("out0", hidden_gpu, cmd) != 0)
    {
        if (error) *error = "Vulkan decoder hidden extract failed";
        return false;
    }
    if (hidden_gpu.empty())
    {
        if (error) *error = "Vulkan decoder hidden output is empty";
        return false;
    }

    VulkanKVCache updated;
    updated.reserve(kTextAttentionLayerCount);
    for (int layer = 0; layer < kTextAttentionLayerCount; ++layer)
    {
        char name_k[32];
        char name_v[32];
        std::snprintf(name_k, sizeof(name_k), "out_cache_k%d", layer);
        std::snprintf(name_v, sizeof(name_v), "out_cache_v%d", layer);
        ncnn::VkMat key;
        ncnn::VkMat value;
        if (ex.extract(name_k, key, cmd) != 0 || ex.extract(name_v, value, cmd) != 0)
        {
            if (error)
            {
                *error = "Vulkan decoder cache extract failed at layer " +
                         std::to_string(layer);
            }
            return false;
        }
        updated.emplace_back(std::move(key), std::move(value));
    }
    cmd.record_download(hidden_gpu, *hidden, option);
    if (cmd.submit_and_wait() != 0)
    {
        if (error) *error = "Vulkan decoder step submission failed";
        return false;
    }

    impl_->caches = std::move(updated);
    ++impl_->step_submits;
    return true;
}

int VulkanDecoderSession::step_submit_count() const
{
    return impl_->step_submits;
}

int VulkanDecoderSession::command_reset_count() const
{
    return impl_->command_resets;
}

int VulkanDecoderSession::input_upload_count() const
{
    return impl_->input_uploads;
}

} // namespace detail
} // namespace hunyuan_ocr

#endif
