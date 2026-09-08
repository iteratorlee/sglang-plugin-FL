"""HYV4 compatibility layer on SGLang's Ascend MLA/NSA/MoE stack."""

import re
from typing import Iterable, Optional, Tuple

import torch
import torch_npu
from torch import nn

import sglang.srt.models.deepseek_v2 as dv
import sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu as npu_moe

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.utils import set_weight_attrs

from .bootstrap import patch_npu_mla_pool_init
from .hc import HYV4HCHeadLayer, HYV4HCLayer
from .hy4_kv_cache import hy4_kv_scatter
from .hy4_triton_projection import (
    hy4_clamped_swiglu,
    hy4_head_bmm,
    hy4_head_bmm_prefill,
)
from .hy4_sparse_attention import hy4_sparse_attention

from . import hy4_triton_moe_gmm as hg


_HYV4_INDEXER_WEIGHT_SUFFIXES = (
    "wq_b.weight",
    "wk.weight",
    "weights_proj.weight",
    "k_norm.weight",
    "k_norm.bias",
)


def _hyv4_full_indexer_layers(config):
    indexer_types = list(getattr(config, "indexer_types", None) or [])
    if len(indexer_types) != config.num_hidden_layers:
        raise RuntimeError(
            f"HYV4 indexer_types has {len(indexer_types)} entries; expected "
            f"num_hidden_layers={config.num_hidden_layers}"
        )
    invalid = sorted(set(indexer_types) - {"full", "shared"})
    if invalid:
        raise RuntimeError(f"HYV4 indexer_types contains unsupported values: {invalid}")
    full_layers = [i for i, kind in enumerate(indexer_types) if kind == "full"]
    if not full_layers or full_layers[0] != 0:
        raise RuntimeError("HYV4 indexer_types must start with a full indexer layer")
    return full_layers


def _hyv4_expected_indexer_weights(config):
    return {
        f"model.layers.{layer_id}.self_attn.indexer.{suffix}"
        for layer_id in _hyv4_full_indexer_layers(config)
        for suffix in _HYV4_INDEXER_WEIGHT_SUFFIXES
    }


def permute_hyv4_indexer_weight(name, loaded_weight, config):
    """Convert checkpoint ``[nope, rope]`` groups to runtime rope-first order.

    The 0.5.11 NPU Indexer splits the first ``qk_rope_head_dim`` channels as
    RoPE channels.  HYV4 checkpoints store those channels at the end of every
    index head for ``wq_b`` and at the end of the single ``wk``/``k_norm``
    group.  This is the ordering used by the official HYV4 loader.
    """
    if ".self_attn.indexer.wq_b." in name:
        group_count = config.index_n_heads
    elif any(
        key in name
        for key in (
            ".self_attn.indexer.wk.",
            ".self_attn.indexer.k_norm.",
        )
    ):
        group_count = 1
    else:
        return loaded_weight

    expected_rows = group_count * config.index_head_dim
    if loaded_weight.ndim < 1 or loaded_weight.shape[0] != expected_rows:
        raise RuntimeError(
            f"HYV4 indexer tensor {name} has shape {tuple(loaded_weight.shape)}; "
            f"expected first dimension {expected_rows}"
        )
    rope_dim = config.qk_rope_head_dim
    if not 0 < rope_dim < config.index_head_dim:
        raise RuntimeError(
            f"HYV4 indexer rope dim {rope_dim} must be within "
            f"index_head_dim={config.index_head_dim}"
        )
    shape = loaded_weight.shape
    grouped = loaded_weight.reshape(
        group_count,
        config.index_head_dim,
        *shape[1:],
    )
    return torch.cat(
        (grouped[:, -rope_dim:], grouped[:, :-rope_dim]), dim=1
    ).reshape(shape)


def _hy_npu_fused_experts_clamped(
    hidden_states,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    top_k,
    **kwargs,
):
    """HYV4 W8A8 routed experts with the checkpoint's SwiGLU limit=10.

    Eager/prefill path. ``npu_moe_compute_expert_tokens`` returns cumulative
    end offsets consumed directly by CANN grouped matmul. Weights are the ND
    int8 cache built in post_load_weights (scale [E, N]); transposed views are
    passed without materializing another model-wide copy.
    """
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx = (
        torch.arange(
            num_tokens * top_k, dtype=torch.int32, device=topk_weights.device
        )
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    ).to(torch.int64)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    # The eager CANN grouped matmul consumes [E,K,N] weights.  A transpose of
    # the retained Triton [E,N,K] ND cache is a zero-copy view and avoids both
    # a 23 GiB model-wide duplicate and the O(tokens*experts) Triton prefill
    # loop. ``expert_tokens`` contains cumulative expert end offsets (type 0).
    g1 = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13.transpose(1, 2)],
        scale=[w13_scale],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=0,
        output_dtype=torch.bfloat16,
    )[0]
    act = hy4_clamped_swiglu(g1)
    # Clamp/SwiGLU changes the activation range, so GEMM2 needs a fresh A8
    # quantization step to preserve the checkpoint's W8A8 contract.
    act_q, act_scale = torch.ops.npu.npu_dynamic_quant(act)
    g2 = torch.ops.npu.npu_grouped_matmul(
        x=[act_q],
        weight=[w2.transpose(1, 2)],
        scale=[w2_scale],
        per_token_scale=[act_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=0,
        output_dtype=torch.bfloat16,
    )[0]
    output = torch.ops.npu.npu_moe_finalize_routing(
        g2,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    return output.view(original_shape) if len(original_shape) == 3 else output


def _hy_npu_fused_experts_w8a8_decode(
    hidden_states,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    top_k,
    **kwargs,
):
    """HYV4 W8A8 decode experts on the Triton grouped-expert GEMM.

    Graph-safe replacement for the stock decode path: the CANN
    ``npu_grouped_matmul`` corrupts values under NPUGraph replay on this stack,
    while the Triton kernels capture and replay value-exactly.  Dynamic A8
    inputs remain INT8 through the dot products; their FP32 per-token scales
    are applied after INT32/FP32 accumulation, matching native eager results
    without an intermediate BF16 activation reconstruction.

    ``init_routing_v2`` (expert_tokens_num_type=1) returns per-expert *counts*,
    so ``et`` is passed to the kernel directly.
    """
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    sorted_hidden, ri, et, ps = torch.ops.npu.npu_moe_init_routing_v2(
        hidden_states,
        topk_ids,
        active_num=num_tokens * top_k,
        expert_num=num_experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_experts],
        quant_mode=1,
    )
    cnt = et.to(torch.int64)
    g1 = hg.hy4_expert_gmm_i8(
        sorted_hidden,
        w13,
        w13_scale,
        cnt,
        pertoken_scale=ps,
    )
    act = hy4_clamped_swiglu(g1)
    act_q, act_scale = torch.ops.npu.npu_dynamic_quant(act)
    g2 = hg.hy4_expert_gmm_i8(
        act_q,
        w2,
        w2_scale,
        cnt,
        pertoken_scale=act_scale,
    )
    output = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=g2, sorted_indices=ri.abs(), probs=topk_weights,
    )
    if len(original_shape) == 3:
        output = output.view(original_shape)
    return output


# --------------------------------------------------------------------------
# Post-PWAL ND conversion for the Triton expert GEMM.
#
# The stock quant method's process_weights_after_loading (which the loader
# runs AFTER load_weights/post_load_weights) transposes the expert weights to
# [E, K, N] and NZ-casts them for npu_grouped_matmul.  The Triton kernel in
# hy4_triton_moe_gmm reads plain ND [E, N, K] with a per-output-row [E, N]
# scale.  Wrap the stock hook: run it first (it also builds the bf16 scale),
# then restore ND with the transpose-if-mismatch rule and drop the NZ copies
# by replacing .data (per-layer, so peak memory stays bounded).
# --------------------------------------------------------------------------
def _hy_pwal_nd_convert(self, layer):
    _HY_ORIG_PWAL(self, layer)
    if not getattr(layer, "_hy4_w8a8", False):
        return
    expected = getattr(layer, "_hy4_expected_layout", None)
    if expected is None:
        raise RuntimeError("HYV4 W8A8 layer is missing its rank-local layout contract")
    for attr, sattr in (
        ("w13_weight", "w13_weight_scale_bf16"),
        ("w2_weight", "w2_weight_scale_bf16"),
    ):
        w = getattr(layer, attr, None)
        s = getattr(layer, sattr, None)
        if w is None or s is None or not hasattr(w, "ndim") or w.ndim != 3:
            raise RuntimeError(f"HYV4 W8A8 missing or invalid {attr}/{sattr}")
        s2 = s.data
        if s2.dim() == 3:
            s2 = s2.squeeze(-1)
            s.data = s2.contiguous()
        if s2.dim() != 2:
            raise RuntimeError(
                f"HYV4 {sattr} must be rank-2 after loading, got {tuple(s2.shape)}"
            )
        try:
            nd = torch_npu.npu_format_cast(w, torch_npu.Format.ND)
            if nd.shape[1] != s2.shape[1]:
                nd = nd.transpose(1, 2)
            nd = nd.contiguous()
            getattr(layer, attr).data = nd
        except Exception as exc:
            raise RuntimeError(f"HYV4 failed to restore {attr} to ND layout") from exc
        if tuple(nd.shape) != expected[attr]:
            raise RuntimeError(
                f"HYV4 {attr} rank-local shape {tuple(nd.shape)} != "
                f"expected {expected[attr]}"
            )
        if tuple(s2.shape) != expected[sattr]:
            raise RuntimeError(
                f"HYV4 {sattr} rank-local shape {tuple(s2.shape)} != "
                f"expected {expected[sattr]}"
            )


def _hy_w8a8_apply(self, layer, dispatch_output):
    """Use HYV4's clamped, graph-safe experts only for marked HYV4 layers."""
    if not getattr(layer, "_hy4_w8a8", False):
        return _HY_ORIG_W8A8_APPLY(self, layer, dispatch_output)

    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    # Release the FP32 copies after the BF16 scale cache is created, matching
    # the stock method's memory lifetime.
    layer.w13_weight_scale = None
    layer.w2_weight_scale = None
    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(hidden_states.dtype)
    fn = (
        _hy_npu_fused_experts_w8a8_decode
        if torch.npu.is_current_stream_capturing()
        else _hy_npu_fused_experts_clamped
    )
    output = fn(
        hidden_states=hidden_states,
        w13=layer.w13_weight,
        w13_scale=layer.w13_weight_scale_bf16,
        w2=layer.w2_weight,
        w2_scale=layer.w2_weight_scale_bf16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=topk_ids.shape[1],
    )
    return StandardCombineInput(hidden_states=output)


_HY_W8A8_METHOD = npu_moe.NPUW8A8Int8DynamicMoEMethod
if not hasattr(_HY_W8A8_METHOD, "_hy4_original_apply"):
    _HY_W8A8_METHOD._hy4_original_apply = _HY_W8A8_METHOD.apply
    _HY_W8A8_METHOD._hy4_original_pwal = (
        _HY_W8A8_METHOD.process_weights_after_loading
    )
_HY_ORIG_W8A8_APPLY = _HY_W8A8_METHOD._hy4_original_apply
_HY_ORIG_PWAL = _HY_W8A8_METHOD._hy4_original_pwal
_HY_W8A8_METHOD.apply = _hy_w8a8_apply
_HY_W8A8_METHOD.process_weights_after_loading = _hy_pwal_nd_convert


# Apply this after the Ascend runtime is initialized (sitecustomize is too
# early in some worker processes).  FlagOS returns this class for NSA as well.
from sglang.srt.hardware_backend.npu.memory_pool_npu import (  # noqa: E402
    NPUMLATokenToKVPool,
)

patch_npu_mla_pool_init(NPUMLATokenToKVPool)

if not hasattr(NPUMLATokenToKVPool, "_hy4_original_set_index_k_buffer"):
    NPUMLATokenToKVPool._hy4_original_set_index_k_buffer = (
        NPUMLATokenToKVPool.set_index_k_buffer
    )

    def _hy4_set_index_k_buffer(self, layer_id, loc, index_k):
        seq_lens = getattr(self, "_hy4_index_scatter_seq_lens", None)
        if seq_lens is None:
            return self._hy4_original_set_index_k_buffer(layer_id, loc, index_k)
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)
        return hy4_kv_scatter(
            self.get_index_k_buffer(layer_id),
            index_k,
            loc,
            seq_lens,
        )

    NPUMLATokenToKVPool.set_index_k_buffer = _hy4_set_index_k_buffer


_original_forward_dsa_core_npu = dv.forward_dsa_core_npu
_original_forward_dsa_prepare_npu = dv.forward_dsa_prepare_npu
_original_forward_mla_core_npu = dv.forward_mla_core_npu
_original_forward_mla_prepare_npu = dv.forward_mla_prepare_npu


def _hy_forward_mla_prepare_npu(
    m,
    positions,
    hidden_states,
    forward_batch,
    zero_allocator,
    layer_scatter_modes,
):
    """Reject accidental HYV4 dense MLA dispatch; preserve other models."""
    if not isinstance(m, HYV4AttentionMLA):
        return _original_forward_mla_prepare_npu(
            m,
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            layer_scatter_modes,
        )
    raise RuntimeError(
        "HYV4 dense MLA dispatch is disabled; native DSA is required"
    )


def _hy_forward_mla_core_npu(
    m,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    forward_batch,
    zero_allocator,
    positions,
    topk_indices,
):
    """Reject accidental HYV4 dense MLA dispatch; preserve other models."""
    if not isinstance(m, HYV4AttentionMLA):
        return _original_forward_mla_core_npu(
            m,
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
        )
    raise RuntimeError(
        "HYV4 dense MLA dispatch is disabled; native DSA is required"
    )


def _hy_forward_dsa_prepare_npu(
    m,
    positions,
    hidden_states,
    forward_batch,
    zero_allocator,
    layer_scatter_modes,
    prev_topk_indices=None,
):
    """Prepare HYV4 DSA queries without capture-unsafe native BMM ops.

    HYV4 needs the normalized Q-LoRA state for the lightning indexer.  The
    bundled DSA prepare path otherwise matches MLA preparation, but its native
    ``batch_matmul_transpose`` decode projection is not replay-safe on this
    CANN release.  Keep variable-token prefill on native BMM and fixed-BS1
    decode on the stride-aware Triton projection validated by this adapter.
    """
    if not isinstance(m, HYV4AttentionMLA):
        return _original_forward_dsa_prepare_npu(
            m,
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            layer_scatter_modes,
            prev_topk_indices,
        )

    q_lora, latent_cache = get_attn_tp_context().fetch_qkv_latent().split(
        [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
    )
    q_lora = m.q_a_layernorm(q_lora)
    k_nope = m.kv_a_layernorm(latent_cache[..., : m.kv_lora_rank]).unsqueeze(1)
    q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
    q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)
    projection = (
        hy4_head_bmm_prefill
        if forward_batch.forward_mode.is_extend()
        else hy4_head_bmm
    )
    q_nope_out = projection(q_nope, m.w_kc)
    if m.layer_id == 0:
        m.rotary_emb.sin_cos_cache = m.rotary_emb.cos_sin_cache.index_select(
            0, positions
        )
    q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

    if m.skip_topk:
        if prev_topk_indices is None:
            raise RuntimeError(
                f"HYV4 shared-index layer {m.layer_id} received no cached top-k"
            )
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


def _hy_forward_dsa_core_npu(
    m,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    topk_indices,
    forward_batch,
    zero_allocator,
    positions,
):
    """Ascend native DSA core with HYV4 sinks and output gate."""
    if not isinstance(m, HYV4AttentionMLA):
        return _original_forward_dsa_core_npu(
            m,
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            topk_indices,
            forward_batch,
            zero_allocator,
            positions,
        )
    pool = forward_batch.token_to_kv_pool
    metadata = forward_batch.attn_backend.forward_metadata
    if topk_indices is None:
        raise RuntimeError(f"HYV4 layer {m.layer_id} DSA received no top-k indices")
    if topk_indices.shape[0] != q_nope_out.shape[0]:
        raise RuntimeError(
            f"HYV4 layer {m.layer_id} top-k rows {topk_indices.shape[0]} != "
            f"query rows {q_nope_out.shape[0]}"
        )
    expected_topk = m.indexer.index_topk
    if topk_indices.reshape(topk_indices.shape[0], -1).shape[1] != expected_topk:
        raise RuntimeError(
            f"HYV4 layer {m.layer_id} top-k width "
            f"{topk_indices.reshape(topk_indices.shape[0], -1).shape[1]} != "
            f"index_topk={expected_topk}"
        )
    if forward_batch.forward_mode.is_extend():
        pool.set_kv_buffer(
            m.attn_mqa,
            forward_batch.out_cache_loc,
            k_nope,
            k_pe,
        )
        cu_seqlens_q = forward_batch.extend_seq_lens.cumsum(
            dim=0, dtype=torch.int32
        )
    else:
        # The stock NPU scatter captures location values instead of reading
        # them again during graph replay.  Reuse the HYV4 device-side scatter
        # so long-context decode appends to the current physical slots.
        hy4_kv_scatter(
            pool.get_key_buffer(m.layer_id),
            k_nope,
            forward_batch.out_cache_loc,
            metadata.seq_lens,
        )
        hy4_kv_scatter(
            pool.get_value_buffer(m.layer_id),
            k_pe,
            forward_batch.out_cache_loc,
            metadata.seq_lens,
        )
        token_count = q_nope_out.shape[0]
        cu_seqlens_q = torch.arange(
            1,
            token_count + 1,
            dtype=torch.int32,
            device=q_nope_out.device,
        )

    attn_output = hy4_sparse_attention(
        query_nope=q_nope_out,
        query_rope=q_pe,
        key_cache=pool.get_key_buffer(m.layer_id),
        rope_cache=pool.get_value_buffer(m.layer_id),
        topk_indices=topk_indices,
        block_table=metadata.block_tables,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens_kv=metadata.seq_lens,
        sink=m.learnable_sink_param,
        scale=m.scaling,
    )
    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)
    projection = (
        hy4_head_bmm_prefill
        if forward_batch.forward_mode.is_extend()
        else hy4_head_bmm
    )
    projected = projection(attn_output, m.w_vc, m._hy_gate_score)
    projected = projected.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(projected)
    return (output, topk_indices) if m.next_skip_topk else (output, None)


# deepseek_v2 imported the NPU helper by value, so replace that binding. The
# wrapper delegates unchanged for every non-HYV4 model.
dv.forward_dsa_core_npu = _hy_forward_dsa_core_npu
dv.forward_dsa_prepare_npu = _hy_forward_dsa_prepare_npu
dv.forward_mla_core_npu = _hy_forward_mla_core_npu
dv.forward_mla_prepare_npu = _hy_forward_mla_prepare_npu


class HYV4Indexer(Indexer):
    """HYV4 Indexer with a device-read graph replay scatter for index K."""

    def forward_npu(
        self,
        x,
        q_lora,
        positions,
        forward_batch,
        layer_id,
        layer_scatter_modes=None,
        dynamic_scale=None,
    ):
        pool = forward_batch.token_to_kv_pool
        use_graph_scatter = not forward_batch.forward_mode.is_extend()
        if use_graph_scatter:
            pool._hy4_index_scatter_seq_lens = (
                forward_batch.attn_backend.forward_metadata.seq_lens
            )
        try:
            return super().forward_npu(
                x,
                q_lora,
                positions,
                forward_batch,
                layer_id,
                layer_scatter_modes,
                dynamic_scale,
            )
        finally:
            if use_graph_scatter:
                del pool._hy4_index_scatter_seq_lens


class HYV4AttentionMLA(dv.DeepseekV2AttentionMLA):
    def dispatch_attn_forward_method(self, forward_batch):
        if getattr(self, "indexer", None) is None:
            raise RuntimeError("HYV4 native DSA requires an instantiated indexer")
        return AttnForwardMethod.DSA_NPU

    def forward(self, positions, hidden_states, forward_batch, zero_allocator, **kwargs):
        gate, _ = self.linear_gate(hidden_states)
        self._hy_gate_score = torch.sigmoid(gate)
        return super().forward(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            **kwargs,
        )


class HYV4MoEGate(dv.MoEGate):
    """HYV4 router GEMM with the FP32 contract used by the reference model."""

    def forward(
        self,
        hidden_states,
        gemm_output_zero_allocator=None,
        forward_batch=None,
    ):
        del gemm_output_zero_allocator, forward_batch
        return torch.nn.functional.linear(hidden_states.float(), self.weight)


class HYV4DecoderLayer(dv.DeepseekV2DecoderLayer):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator,
        gemm_output_zero_allocator=None,
        llama_4_scaling=None,
        prev_topk_indices=None,
    ):
        # iHC owns the residual streams; SGLang's standard residual
        # communicator is intentionally bypassed in this eager correctness path.
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        x, post, hc_residual = self.hc_attn_layer.pre(hidden_states)
        x = self.input_layernorm(x)
        # The stock decoder's LayerCommunicator installs this lazy projection
        # object before entering MLA.  iHC owns the residual/normalization flow,
        # so install only the attention input contract here.
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(x, forward_batch, self.self_attn.prepare_qkv_latent)
        )
        try:
            attn_out = self.self_attn(
                positions=positions,
                hidden_states=x,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                llama_4_scaling=llama_4_scaling,
                prev_topk_indices=prev_topk_indices,
            )
        finally:
            # The context contains layer-local lazy QKV inputs.  Leaving them
            # installed can retain stale storage across a shared-index layer or
            # a graph replay. Newer SGLang exposes clear_attn_inputs(); the
            # bundled 0.5.11 context stores the same state in attn_inputs_.
            attn_context = get_attn_tp_context()
            clear_attn_inputs = getattr(
                attn_context, "clear_attn_inputs", None
            )
            if clear_attn_inputs is not None:
                clear_attn_inputs()
            else:
                attn_context.attn_inputs_ = None
        if isinstance(attn_out, tuple):
            attn_out, topk_indices = attn_out
        else:
            topk_indices = None
        attn_out = tensor_model_parallel_all_reduce(attn_out)
        hidden_states = self.hc_attn_layer.post(attn_out, hc_residual, post)

        hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
        x, post, hc_residual = self.hc_mlp_layer.pre(hidden_states)
        x = self.post_attention_layernorm(x)
        if isinstance(self.mlp, dv.DeepseekV2MLP):
            mlp_out = self.mlp(x)
        else:
            mlp_out = self.mlp(x, forward_batch)
        hidden_states = self.hc_mlp_layer.post(mlp_out, hc_residual, post)
        return hidden_states, None, topk_indices


class HYV4Model(dv.DeepseekV2Model):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if not self.pp_group.is_first_rank or not self.pp_group.is_last_rank:
            raise RuntimeError("HYV4 correctness adapter currently requires pp-size=1")
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )
        zero_allocator = dv.BumpAllocator(
            buffer_size=(self.end_layer - self.start_layer) * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_indices = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, _, topk_indices = layer(
                positions,
                hidden_states,
                forward_batch,
                None,
                zero_allocator,
                None,
                None,
                prev_topk_indices=topk_indices,
            )
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class HYV4ForCausalLM(DeepseekV2ForCausalLM):
    """HYV4 model entry point.

    The core MLA, lightning indexer and MoE tensors share the DeepSeek-v3.2
    layout. HYV4-only tensors are introduced incrementally by this adapter;
    filtering here enables architecture-only and core-weight validation first.
    """

    # Compressed-tensors matches quantization targets against checkpoint
    # projection names.  Both modules below are fused in the SGLang runtime,
    # so declare the source projections explicitly before QuantConfig is built.
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    _deferred_weight_parts = (".mtp_layers.",)

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        self.model.__class__ = HYV4Model
        self.model.config = config
        self.model.hc_head = HYV4HCHeadLayer(
            config, config.hidden_size, config.hc_mult, config.hc_eps
        )
        for layer_id, layer in enumerate(self.model.layers):
            layer.__class__ = HYV4DecoderLayer
            layer.hc_attn_layer = HYV4HCLayer(config, layer_id)
            layer.hc_mlp_layer = HYV4HCLayer(config, layer_id)
            if isinstance(layer.mlp, dv.DeepseekV2MoE):
                # HYV4's GateLinear stores its checkpoint weight in FP32 and
                # emits FP32 logits.  DeepSeek's generic gate follows the
                # model dtype (BF16), which can change the selected top-8
                # experts near routing boundaries.
                layer.mlp.gate.__class__ = HYV4MoEGate
                layer.mlp.gate.weight.data = layer.mlp.gate.weight.data.float()
                # Limit the W8A8 method monkey patch to this HYV4 instance.
                # Non-HYV4 MoE layers delegate to SGLang's original methods.
                layer.mlp.experts._hy4_w8a8 = True
                tp_size = get_tensor_model_parallel_world_size()
                local_moe_width = config.moe_intermediate_size // tp_size
                layer.mlp.experts._hy4_expected_layout = {
                    "w13_weight": (
                        config.n_routed_experts,
                        2 * local_moe_width,
                        config.hidden_size,
                    ),
                    "w13_weight_scale_bf16": (
                        config.n_routed_experts,
                        2 * local_moe_width,
                    ),
                    "w2_weight": (
                        config.n_routed_experts,
                        config.hidden_size,
                        local_moe_width,
                    ),
                    "w2_weight_scale_bf16": (
                        config.n_routed_experts,
                        config.hidden_size,
                    ),
                }
            attn = layer.self_attn
            attn.__class__ = HYV4AttentionMLA
            # FlagGems' generic RoPE bridge indexes the interleaved HYV4 cache
            # with an aclnnIndex path that faults on this CANN/torch-npu stack.
            # Keep the workaround object-local: other models retain the plugin
            # OOT bridge, while HYV4 uses SGLang's native NPU interleave kernel.
            attn.rotary_emb._forward_method = attn.rotary_emb.forward_npu
            # FlagOS is registered as an out-of-tree platform in this image,
            # which makes MultiPlatformOp select Indexer's CUDA signature.
            # The Ascend DSA caller passes NPU-only scatter/dynamic-scale args.
            if getattr(attn, "indexer", None) is None:
                raise RuntimeError(
                    f"HYV4 layer {layer_id} did not construct its native DSA indexer"
                )
            attn.indexer.__class__ = HYV4Indexer
            attn.indexer._forward_method = attn.indexer.forward_npu
            indexer_rope = attn.indexer.rotary_emb
            indexer_rope_forward_npu = getattr(indexer_rope, "forward_npu", None)
            if indexer_rope_forward_npu is None:
                raise RuntimeError(
                    f"HYV4 layer {layer_id} indexer RoPE has no NPU implementation"
                )
            indexer_rope._forward_method = indexer_rope_forward_npu
            attn.linear_gate = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"model.layers.{layer_id}.self_attn.linear_gate",
            )
            attn.learnable_sink_param = nn.Parameter(
                torch.empty(attn.num_local_heads, dtype=torch.float32)
            )
            set_weight_attrs(
                attn.learnable_sink_param,
                {"weight_loader": sharded_weight_loader(0)},
            )

    def determine_num_fused_shared_experts(self, architecture="HYV4ForCausalLM"):
        # Keep shared experts separate on Ascend for direct checkpoint mapping.
        from sglang.srt.server_args import get_global_server_args

        get_global_server_args().disable_shared_experts_fusion = True
        self.num_fused_shared_experts = 0

    def post_load_weights(self, *args, **kwargs):
        """Undo the NPU linear-kernel layout for MLA's absorbed KV weights.

        The generic NPU W8A8 post-loader stores linear weights as [K, N] and
        flattens the channel scale.  DeepSeek's MLA post-loader consumes
        ``kv_b_proj`` once to build ``w_kc``/``w_vc`` and expects the checkpoint
        layout [N, K] with a broadcastable [N, 1] scale.  Restore that logical
        layout before delegating; the original quantized linear is not used by
        the absorbed MLA forward path afterwards.
        """
        for layer in self.model.layers[
            self.model.start_layer : self.model.end_layer
        ]:
            proj = layer.self_attn.kv_b_proj
            weight = getattr(proj, "weight", None)
            if weight is not None and weight.ndim == 2:
                kv_rank = layer.self_attn.kv_lora_rank
                if weight.dtype == torch.int8 and hasattr(proj, "weight_scale"):
                    if weight.shape[0] == kv_rank:
                        # NPU kernel layout [K, N], one scale per N.
                        scale = proj.weight_scale.data.reshape(1, -1).to(
                            torch.bfloat16
                        )
                        restored = (
                            weight.data.to(torch.bfloat16) * scale
                        ).transpose(0, 1).contiguous()
                    elif weight.shape[1] == kv_rank:
                        # Raw checkpoint layout [N, K], one scale per N.  The
                        # real loader invokes model post-processing before the
                        # generic NPU linear transpose; dummy does the reverse.
                        scale = proj.weight_scale.data.reshape(-1, 1).to(
                            torch.bfloat16
                        )
                        restored = (
                            weight.data.to(torch.bfloat16) * scale
                        ).contiguous()
                    else:
                        raise RuntimeError(
                            "HYV4 kv_b_proj has an unsupported rank-local "
                            f"shape {tuple(weight.shape)} for kv rank {kv_rank}"
                        )
                    proj.weight = nn.Parameter(restored, requires_grad=False)
                    proj.weight_scale.data = proj.weight_scale.data.reshape(-1, 1)
                elif weight.shape[0] == kv_rank:
                    weight.data = weight.data.transpose(0, 1).contiguous()
                    if hasattr(proj, "weight_scale"):
                        proj.weight_scale.data = proj.weight_scale.data.reshape(-1, 1)
                elif weight.shape[1] != kv_rank:
                    raise RuntimeError(
                        "HYV4 kv_b_proj has an unsupported unquantized "
                        f"shape {tuple(weight.shape)} for kv rank {kv_rank}"
                    )
        # Routed-expert weights: the ND int8 [E, N, K] conversion now runs in
        # the wrapped process_weights_after_loading (see _hy_pwal_nd_convert),
        # i.e. AFTER the stock NZ transpose, so it sticks.  Nothing to do here.
        out = super().post_load_weights(*args, **kwargs)
        validated = 0
        for layer in self.model.layers[
            self.model.start_layer : self.model.end_layer
        ]:
            attn = layer.self_attn
            h, c, q, v = (
                attn.num_local_heads,
                attn.kv_lora_rank,
                attn.qk_nope_head_dim,
                attn.v_head_dim,
            )
            expected = {
                "w_kc": (h, q, c),
                "w_vc": (h, c, v),
                "linear_gate.weight": (h * v, self.model.config.hidden_size),
                "learnable_sink_param": (h,),
            }
            actual = {
                "w_kc": tuple(attn.w_kc.shape),
                "w_vc": tuple(attn.w_vc.shape),
                "linear_gate.weight": tuple(attn.linear_gate.weight.shape),
                "learnable_sink_param": tuple(attn.learnable_sink_param.shape),
            }
            for name, shape in expected.items():
                if actual[name] != shape:
                    raise RuntimeError(
                        f"HYV4 {name} rank-local shape {actual[name]} != {shape}"
                    )
            validated += 1
        expected_layers = self.model.end_layer - self.model.start_layer
        if validated != expected_layers:
            raise RuntimeError(
                f"HYV4 validated {validated} layers, expected {expected_layers}"
            )

        expected_indexer_weights = _hyv4_expected_indexer_weights(
            self.model.config
        )
        loaded_indexer_weights = getattr(
            self, "_hy4_loaded_indexer_weights", set()
        )
        missing = sorted(expected_indexer_weights - loaded_indexer_weights)
        unexpected = sorted(loaded_indexer_weights - expected_indexer_weights)
        if missing or unexpected:
            raise RuntimeError(
                "HYV4 native DSA indexer checkpoint contract mismatch: "
                f"missing={missing[:8]} ({len(missing)} total), "
                f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
            )

        full_layers = set(_hyv4_full_indexer_layers(self.model.config))
        indexer_validated = 0
        for layer_id, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            indexer = getattr(attn, "indexer", None)
            if indexer is None:
                raise RuntimeError(
                    f"HYV4 layer {layer_id} has no native DSA indexer"
                )
            expected_skip = layer_id not in full_layers
            expected_next_skip = (
                layer_id + 1 < self.model.config.num_hidden_layers
                and layer_id + 1 not in full_layers
            )
            if attn.skip_topk != expected_skip:
                raise RuntimeError(
                    f"HYV4 layer {layer_id} skip_topk={attn.skip_topk} != "
                    f"expected {expected_skip}"
                )
            if attn.next_skip_topk != expected_next_skip:
                raise RuntimeError(
                    f"HYV4 layer {layer_id} next_skip_topk="
                    f"{attn.next_skip_topk} != expected {expected_next_skip}"
                )
            if layer_id not in full_layers:
                continue
            expected_numel = {
                "wq_b.weight": (
                    self.model.config.index_n_heads
                    * self.model.config.index_head_dim
                    * self.model.config.q_lora_rank
                ),
                "wk.weight": (
                    self.model.config.index_head_dim
                    * self.model.config.hidden_size
                ),
                "weights_proj.weight": (
                    self.model.config.index_n_heads
                    * self.model.config.hidden_size
                ),
                "k_norm.weight": self.model.config.index_head_dim,
                "k_norm.bias": self.model.config.index_head_dim,
            }
            actual_tensors = {
                "wq_b.weight": indexer.wq_b.weight,
                "wk.weight": indexer.wk.weight,
                "weights_proj.weight": indexer.weights_proj.weight,
                "k_norm.weight": indexer.k_norm.weight,
                "k_norm.bias": indexer.k_norm.bias,
            }
            for name, tensor in actual_tensors.items():
                if tensor.numel() != expected_numel[name]:
                    raise RuntimeError(
                        f"HYV4 layer {layer_id} indexer {name} numel "
                        f"{tensor.numel()} != {expected_numel[name]}"
                    )
            indexer_validated += 1
        if indexer_validated != len(full_layers):
            raise RuntimeError(
                f"HYV4 validated {indexer_validated} full indexers; expected "
                f"{len(full_layers)}"
            )
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        loaded_indexer_weights = set()

        def core_weights():
            for name, tensor in weights:
                if any(part in name for part in self._deferred_weight_parts):
                    continue
                indexer_match = re.fullmatch(
                    r"model\.layers\.(\d+)\.self_attn\.indexer\.(.+)",
                    name,
                )
                if indexer_match is not None:
                    canonical_name = (
                        f"model.layers.{indexer_match.group(1)}.self_attn."
                        f"indexer.{indexer_match.group(2)}"
                    )
                    loaded_indexer_weights.add(canonical_name)
                    tensor = permute_hyv4_indexer_weight(
                        canonical_name, tensor, self.model.config
                    )
                if name.endswith(".hc_fn") or name.endswith(".hc_head_fn"):
                    name += ".weight"
                yield name, tensor

        self._hy4_loaded_indexer_weights = loaded_indexer_weights
        # ``do_load_weights`` invokes ``post_load_weights`` internally, so
        # publish the mutable set before iteration starts. The generator fills
        # the same object while shards are consumed.
        self.do_load_weights(core_weights(), is_nextn)


EntryClass = HYV4ForCausalLM
