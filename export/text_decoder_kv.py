from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


KVCache = list[tuple[torch.Tensor, torch.Tensor]]
DFLASH_TARGET_LAYER_IDS = (1, 8, 15, 22)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv_for_attention(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return hidden_states
    repeated = []
    for head_index in range(hidden_states.shape[1]):
        head = hidden_states[:, head_index : head_index + 1]
        repeated.extend([head] * num_key_value_groups)
    return torch.cat(repeated, dim=1)


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
        key_states = repeat_kv_for_attention(key_states, self.num_key_value_groups)
        value_states = repeat_kv_for_attention(value_states, self.num_key_value_groups)
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
            enable_gqa=False,
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
        """Preserve the six-input decoder protocol used by existing pnnx exports.

        Transformers 5.13 callers pass the same four-axis mRoPE through both
        input pairs. Keeping the two slots avoids changing the packaged graph IO.
        """
        hidden_states, caches, _ = self.prefill_hf_generate_with_aux(
            inputs_embeds,
            attention_mask,
            xd_cos,
            xd_sin,
            rope_cos,
            rope_sin,
        )
        return hidden_states, caches

    def prefill_hf_generate_with_aux(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        xd_cos: torch.Tensor,
        xd_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, KVCache, tuple[torch.Tensor, ...]]:
        hidden_states = inputs_embeds.to(torch.float32)
        caches: KVCache = []
        auxiliary_hidden_states: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                layer_cos, layer_sin = xd_cos, xd_sin
            else:
                layer_cos, layer_sin = rope_cos, rope_sin
            hidden_states, key_states, value_states = self._layer_forward(
                layer, hidden_states, attention_mask, layer_cos, layer_sin, None, None
            )
            caches.append((key_states, value_states))
            if layer_index in DFLASH_TARGET_LAYER_IDS:
                auxiliary_hidden_states.append(hidden_states)
        if len(auxiliary_hidden_states) != len(DFLASH_TARGET_LAYER_IDS):
            raise RuntimeError(
                f"decoder exposes {len(auxiliary_hidden_states)} DFlash hidden states, "
                f"expected {len(DFLASH_TARGET_LAYER_IDS)}"
            )
        return self.norm(hidden_states), caches, tuple(auxiliary_hidden_states)

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
    ) -> tuple[torch.Tensor, ...]:
        hidden_states, _, auxiliary_hidden_states = self.prefill_hf_generate_with_aux(
            inputs_embeds,
            attention_mask,
            xd_cos,
            xd_sin,
            rope_cos,
            rope_sin,
        )
        return (hidden_states, *auxiliary_hidden_states)
