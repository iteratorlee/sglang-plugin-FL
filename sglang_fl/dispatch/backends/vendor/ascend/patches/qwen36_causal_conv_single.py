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

"""Single-sequence Qwen3.6 causal-convolution prefill on Ascend.

The causal-convolution wrapper bundled in the current NPU image constructs a
Python list and calls ``torch.cat`` even when the packed batch contains exactly
one sequence.  This patch calls the same native convolution directly for the
two TP2 Qwen3.6 channel shapes, preserving the native state writeback and tail
padding semantics.  Everything outside the narrow contract falls back to the
installed wrapper unchanged.
"""

from __future__ import annotations

import functools
import inspect
import logging
import numbers
import sys
from typing import Callable, NamedTuple, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_QWEN36_CONV_SHAPES = frozenset({(4096, 4), (5120, 4)})
_PATCH_MARKER = "_sglang_fl_qwen36_single_sequence"
_BACKEND_MODULE = (
    "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend"
)


class _SingleSequenceMetadata(NamedTuple):
    valid_tokens: int
    cache_index: int
    use_initial_state: bool


def _cpu_sequence_length_matches(
    seq_lens_cpu: Optional[Sequence[int]], valid_tokens: int
) -> bool:
    if seq_lens_cpu is None:
        return True
    try:
        if len(seq_lens_cpu) != 1:
            return False
        seq_len = seq_lens_cpu[0]
    except (IndexError, TypeError, ValueError):
        return False
    return (
        isinstance(seq_len, numbers.Integral)
        and not isinstance(seq_len, bool)
        and seq_len == valid_tokens
    )


def _get_single_sequence_metadata(
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
) -> Optional[_SingleSequenceMetadata]:
    """Return metadata only for the exact semantics-preserving B=1 path."""
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
        or query_start_loc.numel() != 2
        or query_start_loc.dtype not in (torch.int32, torch.int64)
        or query_start_loc.device != x.device
        or cache_indices.ndim != 1
        or cache_indices.numel() != 1
        or cache_indices.dtype not in (torch.int32, torch.int64)
        or cache_indices.device != x.device
        or has_initial_state.ndim != 1
        or has_initial_state.numel() != 1
        or has_initial_state.dtype != torch.bool
        or has_initial_state.device != x.device
        or conv_states.ndim != 3
        or tuple(conv_states.shape[1:]) != (dim, width - 1)
        or conv_states.dtype != x.dtype
        or conv_states.device != x.device
    ):
        return None
    if bias is not None and (
        bias.ndim != 1
        or bias.numel() != dim
        or bias.dtype != x.dtype
        or bias.device != x.device
    ):
        return None

    # The installed wrapper already reads these device scalars in its Python
    # slicing/indexing loop.  B=1 keeps the validation cost bounded.
    try:
        start = int(query_start_loc[0].item())
        valid_tokens = int(query_start_loc[1].item())
        cache_index = int(cache_indices[0].item())
        use_initial_state = bool(has_initial_state[0].item())
    except (RuntimeError, TypeError, ValueError):
        return None

    if (
        start != 0
        or valid_tokens <= 0
        or valid_tokens > x.shape[-1]
        or not _cpu_sequence_length_matches(seq_lens_cpu, valid_tokens)
        or cache_index == pad_slot_id
        or cache_index < 0
        or cache_index >= conv_states.shape[0]
    ):
        return None

    return _SingleSequenceMetadata(
        valid_tokens=valid_tokens,
        cache_index=cache_index,
        use_initial_state=use_initial_state,
    )


def _run_single_sequence(
    native_fn: Callable,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    activation: Optional[str],
    metadata: _SingleSequenceMetadata,
) -> torch.Tensor:
    """Call native once and preserve output/state semantics of the wrapper."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    state = conv_states[metadata.cache_index].unsqueeze(0)
    out, _ = native_fn(
        x[..., : metadata.valid_tokens],
        weight,
        bias,
        initial_states=state if metadata.use_initial_state else None,
        return_final_states=True,
        final_states_out=state,
        activation=activation,
    )

    trailing_tokens = x.shape[-1] - metadata.valid_tokens
    if trailing_tokens:
        out = torch.cat(
            (
                out,
                out.new_zeros([*out.shape[:-1], trailing_tokens]),
            ),
            dim=-1,
        )
    return out


def _make_optimized_causal_conv1d_fn(
    upstream_fn: Callable,
    native_fn: Callable,
) -> Callable:
    @functools.wraps(upstream_fn)
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
        metadata = _get_single_sequence_metadata(
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
            return _run_single_sequence(
                native_fn,
                x,
                weight,
                bias,
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


def _native_api_supported(native_fn: Callable) -> bool:
    try:
        parameters = inspect.signature(native_fn).parameters
    except (TypeError, ValueError):
        return False
    return all(
        name in parameters
        for name in (
            "initial_states",
            "return_final_states",
            "final_states_out",
            "activation",
        )
    )


def patch_qwen36_causal_conv_single() -> bool:
    """Install the guarded B=1 adapter and update imported SGLang aliases."""
    from sgl_kernel_npu.mamba import causal_conv1d as causal_conv

    upstream_fn = causal_conv.causal_conv1d_fn_npu
    if getattr(upstream_fn, _PATCH_MARKER, False):
        return True
    native_fn = causal_conv.causal_conv1d_fn_native
    if not _native_api_supported(native_fn):
        logger.warning(
            "Ascend Qwen3.6 B=1 causal-conv patch skipped: unsupported native API"
        )
        return False

    optimized_fn = _make_optimized_causal_conv1d_fn(upstream_fn, native_fn)
    causal_conv.causal_conv1d_fn_npu = optimized_fn

    backend = sys.modules.get(_BACKEND_MODULE)
    if backend is not None:
        backend.causal_conv1d_fn_npu = optimized_fn
        backend.causal_conv1d_fn = optimized_fn

    logger.info("Ascend Qwen3.6 B=1 causal-conv fast path applied")
    return True
