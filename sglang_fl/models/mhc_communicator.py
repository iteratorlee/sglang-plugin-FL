"""TP-only mHC layer communicator for GLM-5.3 on SGLang v0.5.11."""

from dataclasses import dataclass

import torch

from sglang.srt.distributed import tensor_model_parallel_all_reduce

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
    """Minimal communicator for the requested TP16, DP1 execution mode.

    Attention projections are constructed with ``reduce_results=False`` so an
    explicit TP all-reduce is required before the mHC attention-to-MLP bridge.
    Dense and MoE blocks in v0.5.11 already reduce their own row-parallel
    result, therefore the post-MLP bridge only applies mHC and contracts the
    final layer.
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
        del layer_scatter_modes, allow_reduce_scatter, qkv_latent_func
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
        del forward_batch, cache
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.mhc.combine(hidden_states, residual)
        return self.mhc.split(
            hidden_states, self.mhc.hc_ffn_pre, self.post_attention_layernorm
        )

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        del forward_batch
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
