#include "batch_jsonl.h"

#include "hunyuan_ocr/utf8.h"

#include "picojson.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <utility>

namespace hunyuan_ocr {
namespace detail {
namespace {

struct ParsedRecord {
    BatchRequest request;
    std::string id;
    std::string image;
};

struct BatchPaths {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path input_absolute;
};

void clear_error(RuntimeError* error)
{
    if (error)
    {
        *error = RuntimeError();
    }
}

bool fail(RuntimeError* error, const char* stage, const std::string& message)
{
    if (error)
    {
        error->stage = stage;
        error->message = message;
    }
    return false;
}

bool read_string_field(const picojson::object& object,
                       const char* name,
                       bool required,
                       std::string* value,
                       std::string* message)
{
    const auto found = object.find(name);
    if (found == object.end())
    {
        if (required)
        {
            *message = std::string("missing required field: ") + name;
            return false;
        }
        value->clear();
        return true;
    }
    if (!found->second.is<std::string>())
    {
        *message = std::string("field must be a string: ") + name;
        return false;
    }
    *value = found->second.get<std::string>();
    return true;
}

bool validate_known_fields(const picojson::object& object, std::string* message)
{
    static const std::set<std::string> known = {
        "id", "image", "prompt", "prompt_mode", "max_tokens",
    };
    for (const auto& field : object)
    {
        if (known.find(field.first) == known.end())
        {
            *message = "unknown field: " + field.first;
            return false;
        }
    }
    return true;
}

bool parse_record(const std::string& line,
                  size_t line_number,
                  const std::filesystem::path& input_directory,
                  int default_max_tokens,
                  std::set<std::string>* seen_ids,
                  ParsedRecord* parsed,
                  const char** error_stage,
                  std::string* message)
{
    parsed->request = BatchRequest();
    parsed->request.line = line_number;
    parsed->id.clear();
    parsed->image.clear();
    if (line.empty())
    {
        *error_stage = "batch_parse";
        *message = "input line must be a non-empty JSON object";
        return false;
    }

    picojson::value value;
    const std::string parse_error = picojson::parse(value, line);
    if (!parse_error.empty())
    {
        *error_stage = "batch_parse";
        *message = parse_error;
        return false;
    }
    if (!value.is<picojson::object>())
    {
        *error_stage = "batch_parse";
        *message = "input line must be a non-empty JSON object";
        return false;
    }

    const picojson::object& object = value.get<picojson::object>();
    if (object.empty())
    {
        *error_stage = "batch_parse";
        *message = "input line must be a non-empty JSON object";
        return false;
    }
    *error_stage = "batch_schema";
    if (!validate_known_fields(object, message) ||
        !read_string_field(object, "id", true, &parsed->id, message))
    {
        return false;
    }
    if (parsed->id.empty())
    {
        *message = "id must be a non-empty string";
        return false;
    }
    if (!seen_ids->insert(parsed->id).second)
    {
        *message = "duplicate id: " + parsed->id;
        return false;
    }
    if (!read_string_field(object, "image", true, &parsed->image, message))
    {
        return false;
    }
    if (parsed->image.empty())
    {
        *message = "image must be a non-empty string";
        return false;
    }

    std::string prompt_mode;
    std::string prompt;
    const bool has_prompt_mode = object.find("prompt_mode") != object.end();
    const bool has_prompt = object.find("prompt") != object.end();
    if (has_prompt_mode == has_prompt)
    {
        *message = "exactly one of prompt_mode or prompt is required";
        return false;
    }
    if (has_prompt_mode)
    {
        if (!read_string_field(object, "prompt_mode", true, &prompt_mode, message))
        {
            return false;
        }
        if (prompt_mode == "spotting")
        {
            parsed->request.request.prompt_mode = PromptMode::Spotting;
        }
        else if (prompt_mode == "document")
        {
            parsed->request.request.prompt_mode = PromptMode::Document;
        }
        else
        {
            *message = "prompt_mode must be spotting or document";
            return false;
        }
    }
    else
    {
        if (!read_string_field(object, "prompt", true, &prompt, message))
        {
            return false;
        }
        if (prompt.empty())
        {
            *message = "prompt must be a non-empty string";
            return false;
        }
        parsed->request.request.prompt_mode = PromptMode::Custom;
        parsed->request.request.prompt = std::move(prompt);
    }

    parsed->request.request.max_tokens = default_max_tokens;
    const auto max_tokens = object.find("max_tokens");
    if (max_tokens != object.end())
    {
        if (!max_tokens->second.is<double>())
        {
            *message = "max_tokens must be a positive integer";
            return false;
        }
        const double numeric = max_tokens->second.get<double>();
        if (!std::isfinite(numeric) || numeric <= 0.0 || std::floor(numeric) != numeric ||
            numeric > static_cast<double>(std::numeric_limits<int>::max()))
        {
            *message = "max_tokens must be a positive integer";
            return false;
        }
        parsed->request.request.max_tokens = static_cast<int>(numeric);
    }

    parsed->request.id = parsed->id;
    parsed->request.image = parsed->image;
    const std::filesystem::path image_path = path_from_utf8(parsed->image);
    const std::filesystem::path resolved = image_path.is_absolute()
        ? image_path.lexically_normal()
        : (input_directory / image_path).lexically_normal();
    parsed->request.resolved_image = path_to_utf8(resolved);
    return true;
}

picojson::array token_array(const std::vector<int>& token_ids)
{
    picojson::array tokens;
    tokens.reserve(token_ids.size());
    for (const int token : token_ids)
    {
        tokens.emplace_back(static_cast<double>(token));
    }
    return tokens;
}

std::string serialize_success(const BatchRequest& request, const InferenceResult& result)
{
    picojson::object timing;
    timing["preprocess"] = picojson::value(result.timing.preprocess_ms);
    timing["vision"] = picojson::value(result.timing.vision_ms);
    timing["text"] = picojson::value(result.timing.text_ms);
    timing["total"] = picojson::value(result.timing.total_ms);

    picojson::object output;
    output["line"] = picojson::value(static_cast<double>(request.line));
    output["id"] = picojson::value(request.id);
    output["image"] = picojson::value(request.image);
    output["ok"] = picojson::value(true);
    output["decoder"] = picojson::value(
        result.decoder.mode == DecoderMode::DFlash ? "dflash" : "ar");
    output["text"] = picojson::value(result.text);
    output["token_ids"] = picojson::value(token_array(result.token_ids));
    output["generated_tokens"] = picojson::value(static_cast<double>(result.token_ids.size()));
    output["timing_ms"] = picojson::value(timing);
    if (result.decoder.mode == DecoderMode::DFlash)
    {
        picojson::object dflash;
        dflash["blocks"] = picojson::value(static_cast<double>(result.decoder.block_count));
        dflash["drafted_tokens"] =
            picojson::value(static_cast<double>(result.decoder.drafted_token_count));
        dflash["accepted_tokens"] =
            picojson::value(static_cast<double>(result.decoder.accepted_draft_token_count));
        output["dflash"] = picojson::value(dflash);
    }
    return picojson::value(output).serialize();
}

std::string serialize_failure(size_t line,
                              const std::string& id,
                              const std::string& image,
                              const std::string& stage,
                              const std::string& message)
{
    picojson::object output;
    output["line"] = picojson::value(static_cast<double>(line));
    if (!id.empty()) output["id"] = picojson::value(id);
    if (!image.empty()) output["image"] = picojson::value(image);
    output["ok"] = picojson::value(false);
    output["error_stage"] = picojson::value(stage);
    output["error"] = picojson::value(message);
    return picojson::value(output).serialize();
}

bool write_line(std::ofstream* output, const std::string& line, RuntimeError* error)
{
    *output << line << '\n';
    output->flush();
    if (!output->good())
    {
        return fail(error, "batch_output", "failed to write batch output");
    }
    return true;
}

bool validate_io(const BatchOptions& options, BatchPaths* paths, RuntimeError* error)
{
    if (options.input_path.empty())
    {
        return fail(error, "batch_input", "batch input path must not be empty");
    }
    if (options.output_path.empty())
    {
        return fail(error, "batch_output", "batch output path must not be empty");
    }
    if (options.default_max_tokens <= 0)
    {
        return fail(error, "batch_options", "default max_tokens must be positive");
    }

    BatchPaths local;
    local.input = path_from_utf8(options.input_path);
    local.output = path_from_utf8(options.output_path);
    std::error_code filesystem_error;
    local.input_absolute =
        std::filesystem::absolute(local.input, filesystem_error).lexically_normal();
    if (filesystem_error)
    {
        return fail(error, "batch_input", "failed to resolve batch input path");
    }
    filesystem_error.clear();
    const std::filesystem::path output_absolute =
        std::filesystem::absolute(local.output, filesystem_error).lexically_normal();
    if (filesystem_error)
    {
        return fail(error, "batch_output", "failed to resolve batch output path");
    }
    if (local.input_absolute == output_absolute)
    {
        return fail(error, "batch_output", "batch input and output paths must differ");
    }

    std::ifstream input(local.input, std::ios::binary);
    if (!input.is_open())
    {
        return fail(error, "batch_input", "failed to open batch input: " + options.input_path);
    }
    filesystem_error.clear();
    const bool output_exists = std::filesystem::exists(local.output, filesystem_error);
    if (filesystem_error)
    {
        return fail(error, "batch_output", "failed to inspect batch output path");
    }
    if (output_exists)
    {
        filesystem_error.clear();
        const bool same_file = std::filesystem::equivalent(local.input,
                                                           local.output,
                                                           filesystem_error);
        if (filesystem_error)
        {
            return fail(error, "batch_output", "failed to compare batch input and output paths");
        }
        if (same_file)
        {
            return fail(error, "batch_output", "batch input and output must not identify the same file");
        }
        if (!options.force)
        {
            return fail(error,
                        "batch_output",
                        "batch output already exists; use --force to replace it");
        }
    }

    *paths = std::move(local);
    return true;
}

} // namespace

bool validate_jsonl_batch_io(const BatchOptions& options, RuntimeError* error)
{
    clear_error(error);
    BatchPaths paths;
    return validate_io(options, &paths, error);
}

bool run_jsonl_batch(const BatchOptions& options,
                     const BatchInfer& infer,
                     BatchSummary* summary,
                     RuntimeError* error)
{
    clear_error(error);
    if (summary == nullptr)
    {
        return fail(error, "argument", "batch summary pointer is null");
    }
    *summary = BatchSummary();
    if (!infer)
    {
        return fail(error, "argument", "batch inference callback is empty");
    }
    BatchPaths paths;
    if (!validate_io(options, &paths, error))
    {
        return false;
    }

    std::ifstream input(paths.input, std::ios::binary);
    if (!input.is_open())
    {
        return fail(error, "batch_input", "failed to reopen batch input: " + options.input_path);
    }
    std::ofstream output(paths.output, std::ios::binary | std::ios::trunc);
    if (!output.is_open())
    {
        return fail(error, "batch_output", "failed to open batch output: " + options.output_path);
    }

    std::set<std::string> seen_ids;
    const std::filesystem::path input_directory = paths.input_absolute.parent_path();
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        ++summary->total;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        ParsedRecord parsed;
        const char* record_error_stage = "batch_schema";
        std::string record_error;
        if (!parse_record(line,
                          line_number,
                          input_directory,
                          options.default_max_tokens,
                          &seen_ids,
                          &parsed,
                          &record_error_stage,
                          &record_error))
        {
            ++summary->failed;
            if (!write_line(&output,
                            serialize_failure(line_number,
                                              parsed.id,
                                              parsed.image,
                                              record_error_stage,
                                              record_error),
                            error))
            {
                return false;
            }
            continue;
        }

        InferenceResult result;
        RuntimeError inference_error;
        if (!infer(parsed.request, &result, &inference_error))
        {
            ++summary->failed;
            const std::string stage = inference_error.stage.empty()
                ? "inference"
                : inference_error.stage;
            const std::string message = inference_error.message.empty()
                ? "inference failed without an error message"
                : inference_error.message;
            if (!write_line(&output,
                            serialize_failure(line_number,
                                              parsed.id,
                                              parsed.image,
                                              stage,
                                              message),
                            error))
            {
                return false;
            }
            continue;
        }

        ++summary->succeeded;
        if (!write_line(&output, serialize_success(parsed.request, result), error))
        {
            return false;
        }
    }
    if (input.bad())
    {
        return fail(error, "batch_input", "failed while reading batch input");
    }
    return summary->failed == 0;
}

} // namespace detail
} // namespace hunyuan_ocr
