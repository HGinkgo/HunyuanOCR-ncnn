from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


KVCache = list[tuple[torch.Tensor, torch.Tensor]]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_hunyuan_xdrope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    xdrope_section: Iterable[int],
    rope_theta: float,
    alpha: float,
    cache_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build HF-compatible XD-RoPE cos/sin with shape [1, 1, seq_len, head_dim]."""
    position_ids = position_ids.to(torch.int64)
    device = position_ids.device
    dtype = torch.float32
    seq_len = int(torch.max(position_ids).item()) + 1
    base = float(rope_theta) * float(alpha) ** (float(head_dim) / float(head_dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).float()
    cos_cached = emb.cos().to(cache_dtype).to(torch.float32)
    sin_cached = emb.sin().to(cache_dtype).to(torch.float32)

    x_dim = len(list(xdrope_section))
    section = list(xdrope_section)
    batch = position_ids.shape[0]
    target_len = position_ids.shape[-1]
    output_size = (batch, 1, target_len, target_len)
    cos = cos_cached[position_ids, ...].permute(0, 2, 1, 3).reshape(
        output_size[0], output_size[2], x_dim, -1
    ).contiguous()
    sin = sin_cached[position_ids, ...].permute(0, 2, 1, 3).reshape(
        output_size[0], output_size[2], x_dim, -1
    ).contiguous()

    doubled_section = section * 2
    if sum(doubled_section) != cos.shape[-1]:
        raise ValueError(f"illegal xdrope section: {section}, cos last dim={cos.shape[-1]}")

    cos = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(cos.split(doubled_section, dim=-1))], dim=-1)
    sin = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(sin.split(doubled_section, dim=-1))], dim=-1)
    return cos.reshape(batch, 1, target_len, head_dim), sin.reshape(batch, 1, target_len, head_dim)


def build_hunyuan_1d_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    alpha: float,
    device: torch.device | str = "cpu",
    cache_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build HF-compatible regular 1D RoPE cos/sin with shape [seq_len, head_dim]."""
    dtype = torch.float32
    base = float(rope_theta) * float(alpha) ** (float(head_dim) / float(head_dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).float()
    return emb.cos().to(cache_dtype).to(torch.float32), emb.sin().to(cache_dtype).to(torch.float32)


def apply_external_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x.float() * cos.float()) + (rotate_half(x.float()) * sin.float())


class TextDecoderExternalRopeKVWrapper(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        decoder = copy.deepcopy(decoder).float().eval()
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.num_attention_heads = int(decoder.config.num_attention_heads)
        self.num_key_value_heads = int(decoder.config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = int(decoder.config.head_dim)

    @classmethod
    def from_decoder(cls, decoder: nn.Module) -> "TextDecoderExternalRopeKVWrapper":
        return cls(decoder)

    def _layer_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_in = layer.input_layernorm(hidden_states)
        input_shape = attn_in.shape[:-1]

        query_states = layer.self_attn.q_proj(attn_in).view(*input_shape, self.num_attention_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = layer.self_attn.k_proj(attn_in).view(*input_shape, self.num_key_value_heads, self.head_dim)
        key_states = key_states.transpose(1, 2)
        value_states = layer.self_attn.v_proj(attn_in).view(*input_shape, self.num_key_value_heads, self.head_dim)
        value_states = value_states.transpose(1, 2)

        query_states = apply_external_rope(query_states, cos, sin)
        key_states = apply_external_rope(key_states, cos, sin)
        query_states = layer.self_attn.query_layernorm(query_states)
        key_states = layer.self_attn.key_layernorm(key_states)
        if past_k is not None and past_v is not None:
            key_states = torch.cat((past_k, key_states), dim=2)
            value_states = torch.cat((past_v, value_states), dim=2)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=self.num_key_value_groups != 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = layer.self_attn.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        mlp_in = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(mlp_in)
        return hidden_states, key_states, value_states

    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache]:
        hidden_states = inputs_embeds.to(torch.float32)
        caches: KVCache = []
        for layer in self.layers:
            hidden_states, key_states, value_states = self._layer_forward(
                layer, hidden_states, attention_mask, cos, sin, None, None
            )
            caches.append((key_states, value_states))
        return self.norm(hidden_states), caches

    def prefill_hf_generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        xd_cos: torch.Tensor,
        xd_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache]:
        """Mimic HF generate() cache-prefill RoPE behavior.

        HunyuanVL checks the global DynamicCache length when choosing the RoPE
        branch. During cache prefill, layer 0 sees an empty cache and uses
        XD-RoPE; later layers see layer-0 cache length and use regular 1D RoPE.
        """
        hidden_states = inputs_embeds.to(torch.float32)
        caches: KVCache = []
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                layer_cos, layer_sin = xd_cos, xd_sin
            else:
                layer_cos, layer_sin = rope_cos, rope_sin
            hidden_states, key_states, value_states = self._layer_forward(
                layer, hidden_states, attention_mask, layer_cos, layer_sin, None, None
            )
            caches.append((key_states, value_states))
        return self.norm(hidden_states), caches

    def decode(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        caches: KVCache,
    ) -> tuple[torch.Tensor, KVCache]:
        hidden_states = inputs_embeds.to(torch.float32)
        updated: KVCache = []
        for layer, (past_k, past_v) in zip(self.layers, caches, strict=True):
            hidden_states, key_states, value_states = self._layer_forward(
                layer, hidden_states, attention_mask, cos, sin, past_k, past_v
            )
            updated.append((key_states, value_states))
        return self.norm(hidden_states), updated

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        xd_cos: torch.Tensor,
        xd_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, _ = self.prefill_hf_generate(inputs_embeds, attention_mask, xd_cos, xd_sin, rope_cos, rope_sin)
        return hidden_states
