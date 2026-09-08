"""No-RoPE GLM DSA preparation for the SGLang v0.5.11 NPU path."""

from __future__ import annotations

import torch


def physical_zero_rope(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    storage_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize zero-valued physical RoPE lanes for the Ascend sparse op.

    GLM has no logical rotary dimensions.  CANN 8.5's sparse attention ABI
    still requires the query/key RoPE tensors to use its physical 64-wide
    cache layout.  Expanding empty logical tensors to literal zeros satisfies
    that ABI without adding any rotary dot-product contribution.
    """

    if q_rope.shape[-1] != 0 or k_rope.shape[-1] != 0:
        raise ValueError("physical_zero_rope expects logical zero-width inputs")
    if storage_dim <= 0:
        raise ValueError("physical RoPE storage width must be positive")
    return (
        q_rope.new_zeros((*q_rope.shape[:-1], storage_dim)),
        k_rope.new_zeros((*k_rope.shape[:-1], storage_dim)),
    )


def forward_glm5_dsa_prepare_npu(
    m,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch,
    zero_allocator,
    layer_scatter_modes,
    prev_topk_indices: torch.Tensor = None,
):
    """Prepare GLM MLA/DSA without touching a nonexistent RoPE module.

    GLM-5.3 has ``qk_rope_head_dim == 0``.  The newer NPU implementation has
    a dedicated branch for this case, while v0.5.11 unconditionally reads
    ``m.rotary_emb.is_neox_style``.  This plugin-owned branch keeps the
    v0.5.11 projection/indexer contracts and emits legal empty q/k RoPE views.
    """

    if m.rotary_emb is not None or m.qk_rope_head_dim != 0:
        raise ValueError("GLM-5.3 no-RoPE DSA shim received a RoPE attention layer")

    fused = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
    q, latent_cache = fused.split([m.q_lora_rank, m.kv_lora_rank], dim=-1)
    q_lora = m.q_a_layernorm(q).clone()
    q_nope = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_nope_head_dim)
    k_nope = m.kv_a_layernorm(latent_cache).unsqueeze(1)
    q_pe = q_nope.new_empty((*q_nope.shape[:-1], 0))
    k_pe = latent_cache.new_empty((latent_cache.shape[0], 1, 0))

    q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc).transpose(0, 1)
    if m.skip_topk:
        topk_indices = prev_topk_indices
    else:
        topk_indices = m.indexer(
            hidden_states,
            q_lora,
            positions,
            forward_batch,
            m.layer_id,
            layer_scatter_modes,
            None,
        )

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        topk_indices,
        forward_batch,
        zero_allocator,
        positions,
    )
