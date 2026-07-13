from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_external_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return x.float() * cos.float() + rotate_half(x.float()) * sin.float()


def repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    repeated = []
    for head_index in range(hidden_states.shape[1]):
        head = hidden_states[:, head_index : head_index + 1]
        repeated.extend([head] * groups)
    return torch.cat(repeated, dim=1)


class DFlashExportWrapper(nn.Module):
    """pnnx-friendly HunyuanOCR DFlash draft graph.

    The base decoder supplies four auxiliary hidden states. Token embedding,
    block construction, verification, and lm_head remain outside this graph.
    """

    def __init__(self, draft: nn.Module):
        super().__init__()
        draft = copy.deepcopy(draft).float().eval()
        if len(draft.target_layer_ids) != 4:
            raise ValueError(f"expected four target layers, got {draft.target_layer_ids}")

        self.fc = draft.fc
        self.hidden_norm = draft.hidden_norm
        self.layers = draft.layers
        self.norm = draft.norm
        self.block_size = int(draft.block_size)
        self.hidden_size = int(draft.config.hidden_size)
        self.num_attention_heads = int(draft.config.num_attention_heads)
        self.num_key_value_heads = int(draft.config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = int(draft.config.head_dim)
        self.query_width = self.num_attention_heads * self.head_dim

    @classmethod
    def from_draft(cls, draft: nn.Module) -> "DFlashExportWrapper":
        return cls(draft)

    def _validate_inputs(
        self,
        noise_embedding: torch.Tensor,
        target_states: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if noise_embedding.shape[1] != self.block_size:
            raise ValueError(
                f"noise block must contain {self.block_size} tokens, got {noise_embedding.shape[1]}"
            )
        if noise_embedding.shape[-1] != self.hidden_size:
            raise ValueError(f"noise hidden size must be {self.hidden_size}")
        context_len = target_states[0].shape[1]
        expected_target = (noise_embedding.shape[0], context_len, self.hidden_size)
        if any(tuple(state.shape) != expected_target for state in target_states):
            raise ValueError(f"target hidden states must all have shape {expected_target}")
        total_len = context_len + self.block_size
        expected_rope = (noise_embedding.shape[0], 1, total_len, self.head_dim)
        if tuple(cos.shape) != expected_rope or tuple(sin.shape) != expected_rope:
            raise ValueError(f"cos/sin must both have shape {expected_rope}")
        expected_mask = (noise_embedding.shape[0], 1, self.block_size, total_len)
        if tuple(attention_mask.shape) != expected_mask:
            raise ValueError(f"attention mask must have shape {expected_mask}")

    def _layer_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        attn_in = layer.input_layernorm(hidden_states)
        batch, query_len, _ = attn_in.shape
        context_len = target_hidden.shape[1]

        query = layer.self_attn.q_proj(attn_in).view(
            batch, query_len, self.num_attention_heads, self.head_dim
        )
        query = layer.self_attn.q_norm(query).transpose(1, 2)

        key_context = layer.self_attn.k_proj(target_hidden)
        key_noise = layer.self_attn.k_proj(attn_in)
        value_context = layer.self_attn.v_proj(target_hidden)
        value_noise = layer.self_attn.v_proj(attn_in)
        key = torch.cat((key_context, key_noise), dim=1).view(
            batch,
            context_len + query_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = torch.cat((value_context, value_noise), dim=1).view(
            batch,
            context_len + query_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        key = layer.self_attn.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)

        query = apply_external_rope(query, cos[..., -query_len:, :], sin[..., -query_len:, :])
        key = apply_external_rope(key, cos, sin)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch, query_len, self.query_width
        )
        hidden_states = residual + layer.self_attn.o_proj(attn_output)
        hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return hidden_states

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden_0: torch.Tensor,
        target_hidden_1: torch.Tensor,
        target_hidden_2: torch.Tensor,
        target_hidden_3: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_states = (
            target_hidden_0,
            target_hidden_1,
            target_hidden_2,
            target_hidden_3,
        )
        if not torch.jit.is_tracing():
            self._validate_inputs(noise_embedding, target_states, cos, sin, attention_mask)
        target_hidden = self.hidden_norm(self.fc(torch.cat(target_states, dim=-1)))
        hidden_states = noise_embedding.float()
        for layer in self.layers:
            hidden_states = self._layer_forward(
                layer,
                hidden_states,
                target_hidden,
                cos,
                sin,
                attention_mask,
            )
        return self.norm(hidden_states)
