"""mHC boundaries for attention TP/DP and token-sharded expert parallelism."""

from dataclasses import dataclass

import torch

from sglang.srt.distributed import attention_tensor_model_parallel_all_reduce
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.dp_attention import (
    _dp_gather_via_all_reduce,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
)

from .mhc import hc_contract, hc_expand


@dataclass
class _State:
    hc_mult: int
    hc_attn_pre: object
    hc_ffn_pre: object
    hc_post: object
    h_res: torch.Tensor | None = None
    h_post: torch.Tensor | None = None

    @staticmethod
    def _norm_args(norm):
        if norm is None:
            return None, None
        from sglang.srt.layers.layernorm import RMSNorm

        # Only fuse the native FP32 RMS formula actually selected by this
        # runtime. Other backends may add ModelSlim anti-bias or change casts.
        # Keep calling their own norm instead of silently changing semantics.
        if (
            getattr(norm._forward_method, "__func__", None)
            is not RMSNorm.forward_native
            or norm.cast_x_before_out_mul
            or norm.override_orig_dtype is not None
            or norm.variance_size_override is not None
        ):
            return None, None
        return norm.weight.data, norm.variance_epsilon

    def split(self, x, pre_fn, norm):
        residual = x
        weight, eps = self._norm_args(norm)
        x, self.h_res, self.h_post, fused = pre_fn(x, weight, eps)
        if norm is not None and not fused and x.shape[0]:
            x = norm(x)
        return x, residual

    def combine(self, x, residual):
        return self.hc_post(x, residual, self.h_res, self.h_post)


class MHCLayerCommunicator:
    """Keep mHC residuals local to attention DP, distributing MLP tokens.

    Each attention-TP group reduces its own attention output. Sparse MLPs
    receive a disjoint token slice per rank; their DeepEP result is gathered
    back within that attention group before mHC post-mixing. Dense TP MLPs
    gather/scatter across attention DP using SGLang's padded-token metadata.
    """

    def __init__(
        self,
        layer_scatter_modes,
        input_layernorm,
        post_attention_layernorm,
        allow_reduce_scatter=False,
        is_last_layer=False,
        qkv_latent_func=None,
        *,
        is_first_layer,
        hc_mult,
        hc_attn_pre,
        hc_ffn_pre,
        hc_post,
    ):
        del allow_reduce_scatter, qkv_latent_func
        self.mlp_scattered = layer_scatter_modes.mlp_mode == ScatterMode.SCATTERED
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_dp_size = get_attention_dp_size()
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.mhc = _State(hc_mult, hc_attn_pre, hc_ffn_pre, hc_post)

    def prepare_attn(self, hidden_states, residual, forward_batch):
        del residual, forward_batch
        if self.is_first_layer:
            hidden_states = hc_expand(hidden_states, self.mhc.hc_mult)
        return self.mhc.split(hidden_states, self.mhc.hc_attn_pre, self.input_layernorm)

    def prepare_mlp(self, hidden_states, residual, forward_batch, cache=None):
        del cache
        hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.mhc.combine(hidden_states, residual)
        hidden_states, residual = self.mhc.split(
            hidden_states, self.mhc.hc_ffn_pre, self.post_attention_layernorm
        )
        if self.mlp_scattered:
            tokens = hidden_states.shape[0]
            chunk = (tokens + self.attn_tp_size - 1) // self.attn_tp_size
            start = self.attn_tp_rank * chunk
            valid = max(0, min(chunk, tokens - start))
            if valid == chunk:
                hidden_states = hidden_states.narrow(0, start, chunk)
            else:
                padded = hidden_states.new_zeros((chunk, hidden_states.shape[-1]))
                if valid:
                    padded[:valid].copy_(hidden_states.narrow(0, start, valid))
                hidden_states = padded
        elif self.attn_dp_size > 1:
            gathered = get_global_dp_buffer()
            # Disjoint DP slices use all-reduce for either padding mode.
            # This avoids a reduce-scatter/all-gather pair for MAX_LEN.
            _dp_gather_via_all_reduce(
                gathered, hidden_states, forward_batch, is_partial=False
            )
            hidden_states = gathered
        return hidden_states, residual

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        if self.mlp_scattered and self.attn_tp_size > 1:
            # Every rank in this attention group has the same local length,
            # including an entirely idle group during another DP's prefill.
            if residual.shape[0]:
                chunk = hidden_states.shape[0]
                gathered = hidden_states.new_zeros(
                    (chunk * self.attn_tp_size, hidden_states.shape[-1])
                )
                gathered.narrow(0, self.attn_tp_rank * chunk, chunk).copy_(
                    hidden_states
                )
                # Each output element has only one contributing rank;
                # summing the disjoint slices reconstructs the token batch.
                hidden_states = attention_tensor_model_parallel_all_reduce(gathered)[
                    : residual.shape[0]
                ]
        elif not self.mlp_scattered and self.attn_dp_size > 1:
            local = hidden_states.new_empty(
                (residual.shape[0], hidden_states.shape[-1])
            )
            dp_scatter(local, hidden_states, forward_batch)
            hidden_states = local
        hidden_states = self.mhc.combine(hidden_states, residual)
        self.mhc.h_res = self.mhc.h_post = None
        if self.is_last_layer:
            hidden_states = hc_contract(hidden_states, self.mhc.hc_mult)
        return hidden_states, None

    def maybe_prefetch_next_full_attention_kv(self, *_args, **_kwargs):
        return None

    def should_fuse_mlp_allreduce_with_next_layer(self, _forward_batch):
        return False

    def should_use_reduce_scatter(self, _forward_batch):
        return False


class DSACPLayerCommunicator:
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError("GLM-5.3 plugin supports TP16 without prefill CP")


class MHCHybridDSACPLayerCommunicator(DSACPLayerCommunicator):
    pass
