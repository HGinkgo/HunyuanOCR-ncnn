#include "batch_jsonl.h"

#include "hunyuan_ocr/utf8.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool write_text(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
    return output.good();
}

std::string read_text(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::vector<std::string> read_lines(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line))
    {
        lines.push_back(line);
    }
    return lines;
}

bool contains(const std::string& text, const std::string& needle)
{
    return text.find(needle) != std::string::npos;
}

bool test_ordered_continue_and_escaping(const std::filesystem::path& root)
{
    const std::filesystem::path input = root / "requests.jsonl";
    const std::filesystem::path output = root / "results.jsonl";
    if (!write_text(input,
                    "{\"id\":\"a\",\"image\":\"images/a.png\",\"prompt_mode\":\"document\",\"max_tokens\":7}\n"
                    "{\n"
                    "{\"id\":\"c\",\"image\":\"images/c.png\",\"prompt\":\""
                    "\xE5\x8F\xAA\xE8\xBE\x93\xE5\x87\xBA\xE6\x96\x87\xE5\xAD\x97"
                    "\"}\n"))
    {
        return expect(false, "failed to write ordered input");
    }

    std::vector<std::string> seen_ids;
    hunyuan_ocr::detail::BatchOptions options;
    options.input_path = hunyuan_ocr::path_to_utf8(input);
    options.output_path = hunyuan_ocr::path_to_utf8(output);
    options.default_max_tokens = 128;

    hunyuan_ocr::detail::BatchSummary summary;
    hunyuan_ocr::RuntimeError error;
    const bool ok = hunyuan_ocr::detail::run_jsonl_batch(
        options,
        [&](const hunyuan_ocr::detail::BatchRequest& request,
            hunyuan_ocr::InferenceResult* result,
            hunyuan_ocr::RuntimeError* inference_error) {
            seen_ids.push_back(request.id);
            if (request.id == "a")
            {
                const std::filesystem::path expected = root / "images" / "a.png";
                if (hunyuan_ocr::path_from_utf8(request.resolved_image) != expected ||
                    request.request.prompt_mode != hunyuan_ocr::PromptMode::Document ||
                    request.request.max_tokens != 7)
                {
                    inference_error->stage = "test";
                    inference_error->message = "resolved request mismatch";
                    return false;
                }
                result->text = "first \"line\"\n\xE6\x96\x87\xE6\x9C\xAC";
                result->token_ids = {11, 22};
                result->timing.preprocess_ms = 1.0;
                result->timing.vision_ms = 2.0;
                result->timing.text_ms = 3.0;
                result->timing.total_ms = 6.0;
                return true;
            }
            inference_error->stage = "image_decode";
            inference_error->message = "missing image";
            return false;
        },
        &summary,
        &error);

    const std::vector<std::string> lines = read_lines(output);
    return expect(!ok, "record failures must make batch return false") &&
           expect(error.stage.empty(), "record failures must not set fatal batch error") &&
           expect(seen_ids == std::vector<std::string>({"a", "c"}),
                  "batch callback order mismatch") &&
           expect(summary.total == 3 && summary.succeeded == 1 && summary.failed == 2,
                  "ordered batch summary mismatch") &&
           expect(lines.size() == 3, "ordered batch output line count mismatch") &&
           expect(contains(lines[0], "\"line\":1"), "success line number missing") &&
           expect(contains(lines[0], "\"ok\":true"), "success status missing") &&
           expect(contains(lines[0], "first \\\"line\\\"\\n"), "JSON text escaping mismatch") &&
           expect(contains(lines[0], "\"token_ids\":[11,22]"), "token ids missing") &&
           expect(contains(lines[1], "\"error_stage\":\"batch_parse\""),
                  "parse failure stage missing") &&
           expect(contains(lines[2], "\"id\":\"c\""), "inference failure id missing") &&
           expect(contains(lines[2], "\"error_stage\":\"image_decode\""),
                  "inference failure stage missing");
}

bool test_schema_and_duplicate_ids(const std::filesystem::path& root)
{
    const std::filesystem::path input = root / "invalid.jsonl";
    const std::filesystem::path output = root / "invalid-results.jsonl";
    if (!write_text(input,
                    "{\"id\":\"same\",\"image\":\"a.png\",\"prompt_mode\":\"spotting\"}\n"
                    "{\"id\":\"same\",\"image\":\"b.png\",\"prompt_mode\":\"document\"}\n"
                    "{\"id\":\"both\",\"image\":\"c.png\",\"prompt_mode\":\"document\",\"prompt\":\"x\"}\n"
                    "\n"
                    "{\"id\":\"fraction\",\"image\":\"d.png\",\"prompt_mode\":\"document\",\"max_tokens\":1.5}\n"
                    "{\"id\":\"unknown\",\"image\":\"e.png\",\"prompt_mode\":\"document\",\"threads\":4}\n"))
    {
        return expect(false, "failed to write invalid input");
    }

    int callback_count = 0;
    hunyuan_ocr::detail::BatchOptions options;
    options.input_path = hunyuan_ocr::path_to_utf8(input);
    options.output_path = hunyuan_ocr::path_to_utf8(output);
    hunyuan_ocr::detail::BatchSummary summary;
    hunyuan_ocr::RuntimeError error;
    const bool ok = hunyuan_ocr::detail::run_jsonl_batch(
        options,
        [&](const hunyuan_ocr::detail::BatchRequest&,
            hunyuan_ocr::InferenceResult* result,
            hunyuan_ocr::RuntimeError*) {
            ++callback_count;
            result->text = "ok";
            result->token_ids = {1};
            return true;
        },
        &summary,
        &error);

    const std::vector<std::string> lines = read_lines(output);
    return expect(!ok, "invalid records must fail the batch") &&
           expect(error.stage.empty(), "schema failures must not be fatal") &&
           expect(callback_count == 1, "invalid records reached inference callback") &&
           expect(summary.total == 6 && summary.succeeded == 1 && summary.failed == 5,
                  "invalid schema summary mismatch") &&
           expect(lines.size() == 6, "invalid schema output line count mismatch") &&
           expect(contains(lines[1], "duplicate id"), "duplicate id error missing") &&
           expect(contains(lines[2], "exactly one"), "prompt exclusivity error missing") &&
           expect(contains(lines[3], "non-empty JSON object"), "blank line error missing") &&
           expect(contains(lines[4], "positive integer"), "fractional max_tokens error missing") &&
           expect(contains(lines[5], "unknown field"), "unknown field error missing");
}

bool test_output_policy(const std::filesystem::path& root)
{
    const std::filesystem::path input = root / "force-input.jsonl";
    const std::filesystem::path output = root / "force-output.jsonl";
    if (!write_text(input,
                    "{\"id\":\"force\",\"image\":\"a.png\",\"prompt_mode\":\"document\"}\n") ||
        !write_text(output, "keep\n"))
    {
        return expect(false, "failed to write force fixtures");
    }

    hunyuan_ocr::detail::BatchOptions options;
    options.input_path = hunyuan_ocr::path_to_utf8(input);
    options.output_path = hunyuan_ocr::path_to_utf8(output);
    hunyuan_ocr::detail::BatchSummary summary;
    hunyuan_ocr::RuntimeError error;
    const auto infer = [](const hunyuan_ocr::detail::BatchRequest&,
                          hunyuan_ocr::InferenceResult* result,
                          hunyuan_ocr::RuntimeError*) {
        result->text = "replaced";
        result->token_ids = {7};
        return true;
    };

    if (!expect(!hunyuan_ocr::detail::run_jsonl_batch(options, infer, &summary, &error),
                "existing output must be rejected") ||
        !expect(error.stage == "batch_output", "existing output error stage mismatch") ||
        !expect(read_text(output) == "keep\n", "existing output was modified"))
    {
        return false;
    }

    options.force = true;
    if (!expect(hunyuan_ocr::detail::run_jsonl_batch(options, infer, &summary, &error),
                "force overwrite must succeed") ||
        !expect(summary.total == 1 && summary.succeeded == 1 && summary.failed == 0,
                "force summary mismatch") ||
        !expect(!contains(read_text(output), "keep"), "force did not truncate output"))
    {
        return false;
    }

    options.input_path = hunyuan_ocr::path_to_utf8(input);
    options.output_path = hunyuan_ocr::path_to_utf8(input);
    if (!expect(!hunyuan_ocr::detail::run_jsonl_batch(options, infer, &summary, &error),
                "input and output path equality must fail") ||
        !expect(error.stage == "batch_output", "same-path error stage mismatch"))
    {
        return false;
    }

    const std::filesystem::path hard_link = root / "force-input-hard-link.jsonl";
    std::error_code filesystem_error;
    std::filesystem::create_hard_link(input, hard_link, filesystem_error);
    if (!expect(!filesystem_error, "failed to create input hard link"))
    {
        return false;
    }
    const std::string input_before = read_text(input);
    options.output_path = hunyuan_ocr::path_to_utf8(hard_link);
    return expect(!hunyuan_ocr::detail::run_jsonl_batch(options, infer, &summary, &error),
                  "input hard link used as output must fail") &&
           expect(error.stage == "batch_output", "hard-link error stage mismatch") &&
           expect(read_text(input) == input_before, "input was truncated through hard link");
}

} // namespace

int main()
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "hunyuan_ocr_batch_jsonl_test";
    std::error_code filesystem_error;
    std::filesystem::remove_all(root, filesystem_error);
    filesystem_error.clear();
    std::filesystem::create_directories(root / "images", filesystem_error);
    if (!expect(!filesystem_error, "failed to create batch test directory"))
    {
        return 1;
    }

    const bool ok = test_ordered_continue_and_escaping(root) &&
                    test_schema_and_duplicate_ids(root) &&
                    test_output_policy(root);
    std::filesystem::remove_all(root, filesystem_error);
    if (!ok)
    {
        return 2;
    }

    std::cout << "JSONL batch contract passed\n";
    return 0;
}
