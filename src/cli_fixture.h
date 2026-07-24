#pragma once

#include "cli_options.h"

#include "hunyuan_ocr/text_runtime.h"
#include "hunyuan_ocr/vision_runtime.h"

#include <string>
#include <vector>

namespace hunyuan_ocr::cli {

void print_vision_compute_backend(const VisionRuntimeOptions& options,
                                  const VisionRuntime& runtime);
bool resolve_dynamic_vision_paths(const std::string& model_root,
                                  std::string& param_path,
                                  std::string& bin_path,
                                  std::string& pos_embed_path);
bool print_fixture_vision_diff(const std::vector<float>& actual,
                               const std::string& fixture_dir,
                               std::string& error);
int run_fixture_mode(const CliOptions& options);

} // namespace hunyuan_ocr::cli
