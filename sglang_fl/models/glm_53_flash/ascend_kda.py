"""Ascend compatibility shim for v0.5.11's generic KDA backend.

The generic backend selects an NPU causal-convolution update only for decode;
its prefill convolution and chunk recurrence remain CUDA-oriented Triton
kernels.  This module supplies CANN-compatible prefill/decode paths and maps
graph-padding cache indices to the reserved NPU slot 0.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput


def _physical_npu_conv_state(state: torch.Tensor, kernel_width: int) -> torch.Tensor:
    """Validate and return NPU's physical ``[pool, dim, window]`` cache."""

    expected_window = kernel_width - 1
    if state.ndim != 3 or state.shape[-1] != expected_window:
        raise ValueError(
            "invalid Ascend KDA conv cache layout: expected "
            f"[pool, dim, {expected_window}], got {tuple(state.shape)}"
        )
    return state


def _ascend_kda_decode(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """Run the plugin's CANN-tiled KDA kernel for graph decode.

    SGLang v0.5.11's generic Triton kernel requests more UB than a 910C
    provides at K=V=128.  The image's smaller NPU kernel fits, but implements
    the default softplus gate and ``sqrt(sum(x*x)) + eps`` normalization rather
    than GLM's bounded gate and upstream ``sqrt(sum(x*x) + eps)`` formula.  The
    plugin kernel uses the same BHV=2/BV=64 tiling for prefill and decode.  In
    decode, ``query_start_loc`` describes ``batch`` length-one sequences; this
    helper checks that structural contract before reshaping the flattened gate.
    """

    num_tokens = q.shape[1]
    if (
        k.shape[1] != num_tokens
        or v.shape[1] != num_tokens
        or query_start_loc.numel() != num_tokens + 1
    ):
        raise ValueError(
            "Ascend GLM-5.3 KDA decode requires one varlen sequence per token"
        )
    if v.shape[2] % 2:
        raise ValueError("Ascend GLM-5.3 KDA decode requires an even local head count")

    from .kda_recurrent_npu import glm_kda_varlen_recurrent_npu

    num_value_heads = v.shape[2]
    key_dim = k.shape[-1]
    a = a.reshape(1, num_tokens, num_value_heads, key_dim)
    b = b.reshape(1, num_tokens, num_value_heads)
    return glm_kda_varlen_recurrent_npu(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=ssm_states,
        initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        lower_bound=lower_bound,
    )


def _ascend_kda_prefill_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    state: torch.Tensor,
    has_initial_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Run CANN-native varlen causal convolution with per-request state.

    The image wrapper indexes ``has_initial_state[0]`` for every request.
    That is sufficient for a fresh unchunked batch, but mixes state semantics
    when a continued long request is batched with a new request.  Keep the
    native convolution and update contract while selecting initial state per
    sequence.
    """

    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_native

    starts = [int(value) for value in query_start_loc.cpu().tolist()]
    state_indices = [int(value) for value in cache_indices.cpu().tolist()]
    initial = [bool(value) for value in has_initial_state.cpu().tolist()]
    outputs = []
    for seq_idx, (start, end) in enumerate(zip(starts, starts[1:])):
        state_view = state[state_indices[seq_idx]].unsqueeze(0)
        result, _ = causal_conv1d_fn_native(
            x[..., start:end],
            weight,
            bias,
            activation="silu",
            return_final_states=True,
            final_states_out=state_view,
            initial_states=state_view if initial[seq_idx] else None,
        )
        outputs.append(result)
    output = torch.cat(outputs, dim=-1) if outputs else x[..., :0]
    if output.shape[-1] < x.shape[-1]:
        output = torch.cat(
            (
                output,
                output.new_zeros((*output.shape[:-1], x.shape[-1] - output.shape[-1])),
            ),
            dim=-1,
        )
    return output.transpose(0, 1)


def _ascend_kda_prefill_recurrent(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    seq_lens_cpu,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """Run packed varlen KDA prefill with a 910C-sized recurrent tile.

    The CUDA chunk-KDA kernels in v0.5.11 do not lower on CANN 8.5, while the
    image's NPU recurrent kernel advances the KDA ``a`` pointer incorrectly
    for ``T > 1``.  The plugin variant carries state across the entire packed
    sequence and computes GLM's bounded gate directly.  This removes one
    Python/kernel launch per token and makes long-context prefill practical.
    """

    lengths = [int(length) for length in seq_lens_cpu]
    tensor_tokens = q.shape[1]
    real_tokens = sum(lengths)
    if any(tensor.shape[1] != tensor_tokens for tensor in (k, v, a, b)):
        raise ValueError(
            "Ascend GLM-5.3 KDA prefill requires matching padded token "
            "dimensions for q/k/v/a/b"
        )
    if real_tokens > tensor_tokens:
        raise ValueError(
            "Ascend GLM-5.3 KDA prefill token count mismatch: "
            f"seq_lens={lengths}, tensor_tokens={tensor_tokens}"
        )
    if cache_indices.numel() != len(lengths):
        raise ValueError(
            "Ascend GLM-5.3 KDA prefill requires one cache index per sequence"
        )

    starts = torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=q.device,
    )
    from .kda_recurrent_npu import glm_kda_varlen_recurrent_npu

    output = glm_kda_varlen_recurrent_npu(
        A_log=A_log,
        a=a[:, :real_tokens],
        dt_bias=dt_bias,
        q=q[:, :real_tokens],
        k=k[:, :real_tokens],
        v=v[:, :real_tokens],
        b=b[:, :real_tokens],
        initial_state_source=ssm_states,
        initial_state_indices=cache_indices,
        cu_seqlens=starts,
        lower_bound=lower_bound,
    )

    # SGLang pads extend tokens to the TP alignment.  Do not advance state for
    # padding; preserve the physical tensor shape with a zero tail.
    if real_tokens < tensor_tokens:
        output = torch.cat((output, torch.zeros_like(v[:, real_tokens:])), dim=1)
    return output


class AscendKDAAttnBackend(KDAAttnBackend):
    """KDA backend with legal NPU graph-padding cache indices."""

    def __init__(self, model_runner):
        # This compatibility layer intentionally covers ordinary prefill and
        # decode only.  The optional Ascend speculative-Mamba implementation
        # is not present in the v0.5.11 image and must not be advertised as a
        # silently degraded path.
        if not model_runner.spec_algorithm.is_none():
            raise NotImplementedError(
                "GLM-5.3 Ascend KDA on v0.5.11 does not support speculative "
                "decoding or DFlash"
            )
        server_args = model_runner.server_args
        if not server_args.disable_radix_cache:
            raise ValueError(
                "GLM-5.3 KPool on v0.5.11 requires --disable-radix-cache; "
                "scheduler chunked prefill is supported without prefix reuse"
            )
        chunk_size = server_args.chunked_prefill_size
        if chunk_size > 0 and chunk_size % 128:
            raise ValueError(
                "GLM-5.3 KPool requires --chunked-prefill-size to be a "
                "multiple of 128 so causal prefill tiles remain invariant "
                "across scheduler chunks"
            )
        super().__init__(model_runner)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = _physical_npu_conv_state(
            layer_cache.conv[0], layer.conv_weights.shape[-1]
        )
        cache_indices = self.forward_metadata.mamba_cache_indices

        # Supplying num_accepted_tokens selects sgl_kernel_npu's Triton update
        # path.  The fallback uses advanced indexing whose capture-time cache
        # indices can be baked into an NPU graph.  Slot 0 is graph padding and
        # is explicitly skipped by the Triton kernel.
        num_accepted_tokens = torch.ones(
            (mixed_qkv.shape[0],), dtype=torch.int32, device=mixed_qkv.device
        )
        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            pad_slot_id=0,
        )
        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
        return _ascend_kda_decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=layer_cache.temporal,
            cache_indices=cache_indices,
            query_start_loc=self.forward_metadata.query_start_loc,
            lower_bound=getattr(layer, "lower_bound", None),
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        cache_indices = self.forward_metadata.mamba_cache_indices
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = _physical_npu_conv_state(
            layer_cache.conv[0], layer.conv_weights.shape[-1]
        )
        query_start_loc = self.forward_metadata.query_start_loc
        has_initial_state = forward_batch.extend_prefix_lens > 0
        splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
        q_weight, k_weight, v_weight = layer.conv_weights.split(splits, dim=0)
        q_state, k_state, v_state = conv_states.split(splits, dim=-2)
        if layer.bias is None:
            q_bias, k_bias, v_bias = None, None, None
        else:
            q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)

        # v0.5.11's generic KDA backend deliberately hard-wires the CUDA
        # Triton prefill convolution.  At GLM's local width it requires more
        # unified-buffer storage than a 910C provides.  Use the CANN-native
        # implementation shipped by the same image; unlike the decode update
        # kernel it consumes this physical [pool, dim, window] cache directly.
        def _conv(x, weight, bias, state):
            return _ascend_kda_prefill_conv(
                x,
                weight,
                bias,
                state=state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

        q = _conv(q, q_weight, q_bias, q_state)
        k = _conv(k, k_weight, k_bias, k_state)
        v = _conv(v, v_weight, v_bias, v_state)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
        return _ascend_kda_prefill_recurrent(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=layer_cache.temporal,
            cache_indices=cache_indices,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            lower_bound=getattr(layer, "lower_bound", None),
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = super()._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        # The base implementation uses -1 as a CUDA sentinel.  Request/cache
        # slot 0 is explicitly reserved for graph padding on Ascend.
        metadata.mamba_cache_indices[bs - num_padding :].fill_(0)
        return metadata
