#pragma once

#include <string>
#include <vector>

namespace hunyuan_ocr {

struct ModelFile {
    std::string relative_path;
    bool required;
    std::string note;
};

struct ModelLayoutReport {
    std::string root;
    std::vector<ModelFile> present;
    std::vector<ModelFile> missing_required;
    std::vector<ModelFile> missing_planned;

    bool required_files_present() const;
};

std::vector<ModelFile> expected_model_files();
ModelLayoutReport check_model_layout(const std::string& model_root);

} // namespace hunyuan_ocr

