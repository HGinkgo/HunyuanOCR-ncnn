#include "hunyuan_ocr/model_layout.h"
#include "hunyuan_ocr/utf8.h"

#include <filesystem>
#include <system_error>

namespace hunyuan_ocr {
namespace {

bool regular_file_exists(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::exists(path, error) &&
           std::filesystem::is_regular_file(path, error);
}

} // namespace

bool ModelLayoutReport::required_files_present() const
{
    return missing_required.empty();
}

std::vector<ModelFile> expected_model_files()
{
    return {
        {"model.json", false, "Runtime model manifest; model.json.example documents the current schema."},
        {"tokenizer/vocab.txt", true, "BPE vocabulary exported from the HF tokenizer."},
        {"tokenizer/merges.txt", true, "BPE merge table exported from the HF tokenizer."},
        {"tokenizer/special_tokens.json", true, "Special token list exported from the HF tokenizer."},
        {"tokenizer/eos_ids.json", true, "EOS token ids used by greedy decode."},
        {"text_embed/text_embed.ncnn.param", true, "ncnn text embedding network."},
        {"text_embed/text_embed.ncnn.bin", true, "ncnn text embedding weights."},
        {"text_decoder/text_decoder_kv.ncnn.param", true, "ncnn decoder network with KV cache IO."},
        {"text_decoder/text_decoder_kv.ncnn.bin", true, "ncnn decoder weights."},
        {"lm_head/lm_head.ncnn.param", true, "ncnn lm_head network."},
        {"lm_head/lm_head.ncnn.bin", true, "ncnn lm_head tied embedding weights."},
        {"dflash/dflash.ncnn.param", false, "Optional DFlash draft network."},
        {"dflash/dflash.ncnn.bin", false, "Optional DFlash draft weights."},
        {"vision/vision.ncnn.param", false, "Optional dynamic vision network."},
        {"vision/vision.ncnn.bin", false, "Optional dynamic vision weights."},
        {"vision/pos_embed.bin", false, "Optional dynamic vision base position embedding [1152,128,128]."},
        {"vision/grid_<grid_h>x<grid_w>/vision.ncnn.param", false, "Optional fixed-grid vision network selected from image_grid_thw."},
        {"vision/grid_<grid_h>x<grid_w>/vision.ncnn.bin", false, "Optional fixed-grid vision weights selected from image_grid_thw."},
    };
}

ModelLayoutReport check_model_layout(const std::string& model_root)
{
    ModelLayoutReport report;
    report.root = model_root;

    const std::filesystem::path root_path = path_from_utf8(model_root);
    for (const ModelFile& file : expected_model_files())
    {
        const std::filesystem::path full_path = root_path / file.relative_path;
        if (regular_file_exists(full_path))
        {
            report.present.push_back(file);
        }
        else if (file.required)
        {
            report.missing_required.push_back(file);
        }
        else
        {
            report.missing_planned.push_back(file);
        }
    }

    return report;
}

} // namespace hunyuan_ocr
