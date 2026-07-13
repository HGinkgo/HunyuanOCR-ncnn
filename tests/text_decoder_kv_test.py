#!/usr/bin/env python3

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "export"))

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TextDecoderKvTest(unittest.TestCase):
    def test_repeats_kv_heads_before_sdpa(self) -> None:
        from text_decoder_kv import repeat_kv_for_attention

        values = torch.tensor([[[[1.0]], [[2.0]]]])
        actual = repeat_kv_for_attention(values, 2)

        self.assertEqual(tuple(actual.shape), (1, 4, 1, 1))
        torch.testing.assert_close(actual.flatten(), torch.tensor([1.0, 1.0, 2.0, 2.0]))

    def test_builds_transformers_5_13_mrope_sections_in_fp32(self) -> None:
        from text_decoder_kv import build_hunyuan_mrope_cos_sin

        position_ids = torch.tensor(
            [[[1, 2], [11, 12], [21, 22], [31, 32]]],
            dtype=torch.int64,
        )
        cos, sin = build_hunyuan_mrope_cos_sin(
            position_ids,
            head_dim=128,
            mrope_section=[16, 16, 16, 16],
            rope_theta=10000.0,
            alpha=1000.0,
        )

        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)
        self.assertEqual(tuple(cos.shape), (1, 1, 2, 128))

        base = 10000.0 * 1000.0 ** (128.0 / 126.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128.0))
        frequencies = torch.cat((inv_freq, inv_freq))
        expected_positions = [1, 11, 21, 31]
        for section_index, position in enumerate(expected_positions):
            start = section_index * 32
            stop = start + 32
            expected = torch.cos(position * frequencies[start:stop])
            torch.testing.assert_close(cos[0, 0, 0, start:stop], expected)

    def test_forward_keeps_final_hidden_as_out0_and_exposes_dflash_layers(self) -> None:
        from text_decoder_kv import DFLASH_TARGET_LAYER_IDS, TextDecoderExternalRopeKVWrapper

        class FakeDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)])
                self.norm = torch.nn.Identity()
                self.config = SimpleNamespace(
                    num_attention_heads=1,
                    num_key_value_heads=1,
                    head_dim=1,
                )

        wrapper = TextDecoderExternalRopeKVWrapper(FakeDecoder())

        def fake_layer_forward(
            self,
            layer,
            hidden_states,
            attention_mask,
            cos,
            sin,
            past_k,
            past_v,
        ):
            del self, layer, attention_mask, cos, sin, past_k, past_v
            updated = hidden_states + 1.0
            cache = torch.empty(0)
            return updated, cache, cache

        wrapper._layer_forward = types.MethodType(fake_layer_forward, wrapper)
        dummy = torch.zeros(1, 1, 1)
        outputs = wrapper(dummy, dummy, dummy, dummy, dummy, dummy)

        self.assertEqual(DFLASH_TARGET_LAYER_IDS, (1, 8, 15, 22))
        self.assertEqual(len(outputs), 5)
        torch.testing.assert_close(outputs[0], torch.full_like(dummy, 24.0))
        for output, expected in zip(outputs[1:], (2.0, 9.0, 16.0, 23.0)):
            torch.testing.assert_close(output, torch.full_like(dummy, expected))

        legacy_hidden, caches = wrapper.prefill_hf_generate(dummy, dummy, dummy, dummy, dummy, dummy)
        torch.testing.assert_close(legacy_hidden, torch.full_like(dummy, 24.0))
        self.assertEqual(len(caches), 24)


if __name__ == "__main__":
    unittest.main()
