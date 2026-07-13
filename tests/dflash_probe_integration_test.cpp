#include "hunyuan_ocr/text_runtime.h"

#include <iostream>
#include <numeric>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: dflash_probe_integration_test MODEL_ROOT FIXTURE_DIR\n";
        return 1;
    }

    std::string error;
    hunyuan_ocr::TextRuntime runtime(4);
    if (!runtime.load(argv[1], &error) || !runtime.load_dflash(argv[1], &error))
    {
        std::cerr << error << '\n';
        return 2;
    }

    hunyuan_ocr::DFlashBlockProbeResult result;
    if (!runtime.run_vlm_fixture_dflash_probe(argv[2], &result, &error))
    {
        std::cerr << error << '\n';
        return 3;
    }
    if (!result.first_token_matches_expected ||
        !result.first_target_token_matches_expected)
    {
        return 4;
    }
    if (result.proposed_tokens.size() != 16 ||
        result.target_tokens.size() != 16 ||
        result.acceptance_length < 0 ||
        result.acceptance_length > 15)
    {
        return 5;
    }
    if (result.correction_token !=
        result.target_tokens[static_cast<size_t>(result.acceptance_length)])
    {
        return 6;
    }

    std::cout << "first_token=" << result.first_token << '\n';
    std::cout << "acceptance_length=" << result.acceptance_length << '\n';
    std::cout << "correction_token=" << result.correction_token << '\n';
    std::cout << "proposed_tokens=";
    for (int token : result.proposed_tokens)
    {
        std::cout << token << ',';
    }
    std::cout << '\n';
    std::cout << "target_tokens=";
    for (int token : result.target_tokens)
    {
        std::cout << token << ',';
    }
    std::cout << '\n';

    hunyuan_ocr::DFlashDecodeResult decode;
    if (!runtime.run_vlm_fixture_dflash_decode(argv[2], 30, &decode, &error))
    {
        std::cerr << error << '\n';
        return 7;
    }
    const int accepted = std::accumulate(
        decode.acceptance_lengths.begin(), decode.acceptance_lengths.end(), 0);
    if (!decode.decode.matches_expected() ||
        decode.decode.generated_tokens.size() != 30 ||
        decode.block_count != 11 ||
        decode.drafted_token_count != decode.block_count * 15 ||
        decode.accepted_draft_token_count != accepted ||
        decode.acceptance_lengths.back() != 0)
    {
        return 8;
    }

    const hunyuan_ocr::DFlashDecodeTiming& timing = decode.timing;
    if (timing.total_ms <= 0.0 ||
        timing.prefill_ms <= 0.0 ||
        timing.draft_infer_ms <= 0.0 ||
        timing.verify_infer_ms <= 0.0)
    {
        return 9;
    }
    if (timing.prefill_ms < 0.0 ||
        timing.draft_prepare_ms < 0.0 ||
        timing.draft_infer_ms < 0.0 ||
        timing.draft_postprocess_ms < 0.0 ||
        timing.verify_prepare_ms < 0.0 ||
        timing.verify_infer_ms < 0.0 ||
        timing.verify_postprocess_ms < 0.0 ||
        timing.commit_ms < 0.0 ||
        timing.total_ms < 0.0)
    {
        return 10;
    }
    const double stage_sum = timing.prefill_ms + timing.draft_prepare_ms +
                             timing.draft_infer_ms + timing.draft_postprocess_ms +
                             timing.verify_prepare_ms + timing.verify_infer_ms +
                             timing.verify_postprocess_ms + timing.commit_ms;
    if (stage_sum > timing.total_ms * 1.05)
    {
        return 11;
    }

    std::cout << "decode_blocks=" << decode.block_count << '\n';
    std::cout << "decode_accepted=" << decode.accepted_draft_token_count << '\n';
    std::cout << "timing_total_ms=" << timing.total_ms << '\n';
    std::cout << "timing_prefill_ms=" << timing.prefill_ms << '\n';
    std::cout << "timing_draft_infer_ms=" << timing.draft_infer_ms << '\n';
    std::cout << "timing_verify_infer_ms=" << timing.verify_infer_ms << '\n';
    return 0;
}
