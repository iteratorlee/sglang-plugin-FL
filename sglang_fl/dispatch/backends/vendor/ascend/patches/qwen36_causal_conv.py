# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch equal-length Qwen3.6 causal-convolution prefill on Ascend.

The current ``sgl-kernel-npu`` packed-input wrapper invokes the native
convolution once per sequence and concatenates the outputs.  SGLang's normal
chunked-prefill batches contain equal-length sequences (for example 16x1K or
4x4K), so they can be reshaped into one dense batch and processed by the same
native implementation in one call.

This patch deliberately does not replace the convolution arithmetic.  It only
removes per-sequence Python dispatch and intermediate outputs.  Unsupported
shapes, mixed initial-state batches, padded cache slots, and unequal sequence
lengths retain the upstream implementation.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from numbers import Integral
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_QWEN36_CONV_SHAPES = frozenset({(4096, 4), (5120, 4)})
_PATCH_MARKER = "_sglang_fl_qwen36_equal_length_batched"


class _BatchMetadata(NamedTuple):
    batch_size: int
    sequence_length: int
    valid_tokens: int
    use_initial_state: bool


def _get_batch_metadata(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    activation: Optional[str],
    pad_slot_id: int,
    seq_lens_cpu: Optional[Sequence[int]] = None,
) -> Optional[_BatchMetadata]:
    """Return metadata only for the narrow, semantics-preserving fast path."""
    if (
        x.ndim != 2
        or weight.ndim != 2
        or tuple(weight.shape) not in _QWEN36_CONV_SHAPES
        or x.shape[0] != weight.shape[0]
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or x.device != weight.device
        or activation not in (None, "silu", "swish")
        or query_start_loc is None
        or cache_indices is None
        or has_initial_state is None
        or conv_states is None
    ):
        return None

    dim, width = weight.shape
    if (
        query_start_loc.ndim != 1
        or cache_indices.ndim != 1
        or has_initial_state.ndim != 1
        or conv_states.ndim != 3
        or tuple(conv_states.shape[1:]) != (dim, width - 1)
        or conv_states.dtype != x.dtype
        or conv_states.device != x.device
        or query_start_loc.device != x.device
        or cache_indices.device != x.device
        or has_initial_state.device != x.device
        or has_initial_state.dtype != torch.bool
        or cache_indices.dtype not in (torch.int32, torch.int64)
        or query_start_loc.dtype not in (torch.int32, torch.int64)
    ):
        return None
    if bias is not None and (
        bias.ndim != 1
        or bias.numel() != dim
        or bias.dtype != x.dtype
        or bias.device != x.device
    ):
        return None

    batch_size = query_start_loc.numel() - 1
    if (
        batch_size <= 1
        or cache_indices.numel() != batch_size
        or has_initial_state.numel() != batch_size
    ):
        return None

    if seq_lens_cpu is not None:
        # SGLang 0.5.11 passes ForwardBatch.extend_seq_lens_cpu alongside
        # query_start_loc.  Prefer this authoritative CPU mirror so every GDN
        # layer does not repeat device sub/eq/all plus scalar synchronizations.
        if (
            not isinstance(seq_lens_cpu, Sequence)
            or isinstance(seq_lens_cpu, (str, bytes))
            or len(seq_lens_cpu) != batch_size
            or not all(
                isinstance(length, Integral) and not isinstance(length, bool)
                for length in seq_lens_cpu
            )
        ):
            return None
        sequence_length = int(seq_lens_cpu[0])
        if sequence_length <= 0 or not all(
            int(length) == sequence_length for length in seq_lens_cpu
        ):
            return None
        valid_tokens = batch_size * sequence_length
        if valid_tokens > x.shape[-1]:
            return None
    else:
        # Keep the strict device-metadata path for callers without the SGLang
        # CPU mirror.  It is slower, but retains the original guarded behavior.
        sequence_lengths = query_start_loc[1:] - query_start_loc[:-1]
        sequence_length = int(sequence_lengths[0].item())
        if sequence_length <= 0 or not bool(
            torch.all(sequence_lengths == sequence_length).item()
        ):
            return None

        start = int(query_start_loc[0].item())
        valid_tokens = int(query_start_loc[-1].item())
        if (
            start != 0
            or valid_tokens != batch_size * sequence_length
            or valid_tokens > x.shape[-1]
        ):
            return None

    if bool(torch.any(cache_indices == pad_slot_id).item()):
        return None

    use_initial_state = bool(has_initial_state[0].item())
    if not bool(torch.all(has_initial_state == use_initial_state).item()):
        # The installed upstream wrapper uses has_initial_state[0] for every
        # row.  Do not change that behavior for a mixed-state batch.
        return None

    return _BatchMetadata(
        batch_size=batch_size,
        sequence_length=sequence_length,
        valid_tokens=valid_tokens,
        use_initial_state=use_initial_state,
    )


def _run_equal_length_batched(
    native_fn: Callable,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cache_indices: torch.Tensor,
    conv_states: torch.Tensor,
    activation: Optional[str],
    metadata: _BatchMetadata,
) -> torch.Tensor:
    """Execute one dense batched native convolution and repack its output."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    dim = x.shape[0]
    dense_x = (
        x[:, : metadata.valid_tokens]
        .view(dim, metadata.batch_size, metadata.sequence_length)
        .transpose(0, 1)
        .contiguous()
    )
    initial_states = (
        torch.index_select(conv_states, 0, cache_indices)
        if metadata.use_initial_state
        else None
    )
    dense_out, final_states = native_fn(
        dense_x,
        weight,
        bias,
        initial_states=initial_states,
        return_final_states=True,
        activation=activation,
    )
    conv_states.index_copy_(0, cache_indices, final_states)

    packed_out = (
        dense_out.transpose(0, 1)
        .contiguous()
        .view(dim, metadata.valid_tokens)
    )
    trailing_tokens = x.shape[-1] - metadata.valid_tokens
    if trailing_tokens:
        packed_out = F.pad(packed_out, (0, trailing_tokens))
    return packed_out


def _make_optimized_causal_conv1d_fn(
    upstream_fn: Callable,
    native_fn: Callable,
) -> Callable:
    def optimized_causal_conv1d_fn_npu(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        cache_indices: Optional[torch.Tensor] = None,
        has_initial_state: Optional[torch.Tensor] = None,
        conv_states: Optional[torch.Tensor] = None,
        activation: Optional[str] = "silu",
        pad_slot_id: int = -1,
        **kwargs,
    ) -> torch.Tensor:
        metadata = _get_batch_metadata(
            x,
            weight,
            bias,
            query_start_loc,
            cache_indices,
            has_initial_state,
            conv_states,
            activation,
            pad_slot_id,
            kwargs.get("seq_lens_cpu"),
        )
        if metadata is not None:
            return _run_equal_length_batched(
                native_fn,
                x,
                weight,
                bias,
                cache_indices,
                conv_states,
                activation,
                metadata,
            )
        return upstream_fn(
            x,
            weight,
            bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_states,
            activation=activation,
            pad_slot_id=pad_slot_id,
            **kwargs,
        )

    setattr(optimized_causal_conv1d_fn_npu, _PATCH_MARKER, True)
    return optimized_causal_conv1d_fn_npu


def patch_qwen36_causal_conv_prefill() -> None:
    """Install the guarded batching adapter and preserve imported aliases."""
    from sgl_kernel_npu.mamba import causal_conv1d as causal_conv

    upstream_fn = causal_conv.causal_conv1d_fn_npu
    if getattr(upstream_fn, _PATCH_MARKER, False):
        return

    optimized_fn = _make_optimized_causal_conv1d_fn(
        upstream_fn,
        causal_conv.causal_conv1d_fn_native,
    )
    causal_conv.causal_conv1d_fn_npu = optimized_fn

    # SGLang imports the wrapper by value, so update it as well when the
    # attention backend was initialized before the plugin patch ran.
    backend = sys.modules.get(
        "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend"
    )
    if backend is not None:
        backend.causal_conv1d_fn_npu = optimized_fn
        backend.causal_conv1d_fn = optimized_fn

    logger.info("Ascend Qwen3.6 equal-length batched causal-conv path applied")
