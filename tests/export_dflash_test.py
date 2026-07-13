#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "export"))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None


TORCH_SDPA_AVAILABLE = torch is not None and hasattr(F, "scaled_dot_product_attention")


if torch is not None:
    class _RMSNorm(nn.Module):
        def __init__(self, size: int, eps: float = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(size))
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)


    class _ToyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 4
            self.num_key_value_groups = 2
            self.scaling = self.head_dim**-0.5
            self.q_proj = nn.Linear(8, 16, bias=False)
            self.k_proj = nn.Linear(8, 8, bias=False)
            self.v_proj = nn.Linear(8, 8, bias=False)
            self.o_proj = nn.Linear(16, 8, bias=False)
            self.q_norm = _RMSNorm(4)
            self.k_norm = _RMSNorm(4)


    class _ToyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)
            self.up_proj = nn.Linear(8, 16, bias=False)
            self.down_proj = nn.Linear(16, 8, bias=False)
            self.act_fn = F.silu

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


    class _ToyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttention()
            self.mlp = _ToyMLP()
            self.input_layernorm = _RMSNorm(8)
            self.post_attention_layernorm = _RMSNorm(8)


    class _ToyDraft(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
            )
            self.target_layer_ids = [1, 8, 15, 22]
            self.block_size = 2
            self.fc = nn.Linear(32, 8, bias=False)
            self.hidden_norm = _RMSNorm(8)
            self.layers = nn.ModuleList([_ToyLayer()])
            self.norm = _RMSNorm(8)


class DFlashExportShapeTest(unittest.TestCase):
    def test_pnnx_shapes_keep_block_fixed_and_context_dynamic(self) -> None:
        from _common import pnnx_inputshape, pnnx_intermediate_paths

        self.assertEqual(
            pnnx_inputshape(context_len=8, block_size=16, hidden_size=1024, head_dim=128),
            "[1,16,1024]f32,[1,8,1024]f32,[1,8,1024]f32,[1,8,1024]f32,"
            "[1,8,1024]f32,[1,1,24,128]f32,[1,1,24,128]f32,[1,1,16,24]f32",
        )
        self.assertEqual(
            [path.name for path in pnnx_intermediate_paths(Path("dflash.pt"))],
            [
                "dflash.pt",
                "dflash.pnnx.param",
                "dflash.pnnx.bin",
                "dflash.pnnx.onnx",
                "dflash_pnnx.py",
                "dflash_ncnn.py",
            ],
        )


@unittest.skipUnless(TORCH_SDPA_AVAILABLE, "PyTorch with SDPA support is required")
class DFlashExportTest(unittest.TestCase):
    def test_wrapper_matches_dflash_attention_contract(self) -> None:
        from dflash_export import DFlashExportWrapper

        torch.manual_seed(7)
        draft = _ToyDraft().eval()
        wrapper = DFlashExportWrapper.from_draft(draft).eval()
        target_states = [torch.randn(1, 3, 8) for _ in range(4)]
        noise = torch.randn(1, 2, 8)
        angles = torch.randn(1, 1, 5, 4)
        cos = angles.cos()
        sin = angles.sin()
        mask = torch.zeros(1, 1, 2, 5)

        actual = wrapper(noise, *target_states, cos, sin, mask)
        expected = self._reference(draft, noise, target_states, cos, sin, mask)

        self.assertEqual(tuple(actual.shape), (1, 2, 8))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_wrapper_rejects_non_block_sized_noise(self) -> None:
        from dflash_export import DFlashExportWrapper

        wrapper = DFlashExportWrapper.from_draft(_ToyDraft()).eval()
        targets = [torch.zeros(1, 3, 8) for _ in range(4)]
        with self.assertRaisesRegex(ValueError, "noise block"):
            wrapper(
                torch.zeros(1, 1, 8),
                *targets,
                torch.ones(1, 1, 4, 4),
                torch.zeros(1, 1, 4, 4),
                torch.zeros(1, 1, 1, 4),
            )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    @classmethod
    def _reference(
        cls,
        draft: _ToyDraft,
        noise: torch.Tensor,
        target_states: list[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = draft.hidden_norm(draft.fc(torch.cat(target_states, dim=-1)))
        hidden = noise
        for layer in draft.layers:
            residual = hidden
            attn_in = layer.input_layernorm(hidden)
            q = layer.self_attn.q_proj(attn_in).view(1, 2, 4, 4).transpose(1, 2)
            k_ctx = layer.self_attn.k_proj(target)
            k_noise = layer.self_attn.k_proj(attn_in)
            v_ctx = layer.self_attn.v_proj(target)
            v_noise = layer.self_attn.v_proj(attn_in)
            k = torch.cat((k_ctx, k_noise), dim=1).view(1, 5, 2, 4).transpose(1, 2)
            v = torch.cat((v_ctx, v_noise), dim=1).view(1, 5, 2, 4).transpose(1, 2)
            q = layer.self_attn.q_norm(q)
            k = layer.self_attn.k_norm(k)
            q = q * cos[..., -2:, :] + cls._rotate_half(q) * sin[..., -2:, :]
            k = k * cos + cls._rotate_half(k) * sin
            k = k.repeat_interleave(2, dim=1)
            v = v.repeat_interleave(2, dim=1)
            hidden = F.scaled_dot_product_attention(q, k, v, mask, scale=0.5)
            hidden = hidden.transpose(1, 2).reshape(1, 2, 16)
            hidden = residual + layer.self_attn.o_proj(hidden)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return draft.norm(hidden)


if __name__ == "__main__":
    unittest.main()
