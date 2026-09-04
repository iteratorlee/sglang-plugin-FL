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

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches import (
    qwen36_causal_conv as conv_batched,
)
from sglang_fl.dispatch.backends.vendor.ascend.patches import (
    qwen36_causal_conv_single as conv_single,
)


def _inputs(batch_size: int, dim: int = 4096, sequence_length: int = 3):
    valid_tokens = batch_size * sequence_length
    return {
        "x": torch.arange(
            dim * valid_tokens, dtype=torch.bfloat16
        ).view(dim, valid_tokens),
        "weight": torch.ones(dim, 4, dtype=torch.bfloat16),
        "query_start_loc": torch.arange(
            0,
            valid_tokens + 1,
            sequence_length,
            dtype=torch.int32,
        ),
        "cache_indices": torch.arange(batch_size, dtype=torch.int64),
        "has_initial_state": torch.zeros(batch_size, dtype=torch.bool),
        "conv_states": torch.ones(
            batch_size, dim, 3, dtype=torch.bfloat16
        ),
        "seq_lens_cpu": [sequence_length] * batch_size,
    }


def _invoke(fn, inputs):
    return fn(
        inputs["x"],
        inputs["weight"],
        query_start_loc=inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        has_initial_state=inputs["has_initial_state"],
        conv_states=inputs["conv_states"],
        activation="silu",
        pad_slot_id=-1,
        seq_lens_cpu=inputs["seq_lens_cpu"],
    )


def test_batched_then_single_composition_order_routing_and_idempotence():
    import sgl_kernel_npu.mamba as mamba

    stock_calls = []
    native_calls = []
    stock_sentinel = object()

    def stock(*args, **kwargs):
        stock_calls.append((args, kwargs))
        return stock_sentinel

    def native(
        x,
        weight,
        bias=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation="silu",
    ):
        batch_size = 1 if x.ndim == 2 else x.shape[0]
        final_states = torch.zeros(
            batch_size,
            weight.shape[0],
            weight.shape[1] - 1,
            dtype=x.dtype,
            device=x.device,
        )
        if final_states_out is not None:
            final_states_out.copy_(final_states)
            final_states = final_states_out
        native_calls.append(
            {
                "ndim": x.ndim,
                "batch_size": batch_size,
                "initial_states": initial_states,
                "return_final_states": return_final_states,
                "activation": activation,
            }
        )
        return x, final_states

    causal_conv = SimpleNamespace(
        causal_conv1d_fn_npu=stock,
        causal_conv1d_fn_native=native,
    )
    backend = SimpleNamespace(
        causal_conv1d_fn_npu=stock,
        causal_conv1d_fn=stock,
    )
    with (
        mock.patch.object(mamba, "causal_conv1d", causal_conv, create=True),
        mock.patch.dict(
            sys.modules, {conv_single._BACKEND_MODULE: backend}
        ),
    ):
        conv_batched.patch_qwen36_causal_conv_prefill()
        batched_wrapper = causal_conv.causal_conv1d_fn_npu
        assert getattr(batched_wrapper, conv_batched._PATCH_MARKER)
        assert not getattr(
            batched_wrapper, conv_single._PATCH_MARKER, False
        )
        assert backend.causal_conv1d_fn_npu is batched_wrapper
        assert backend.causal_conv1d_fn is batched_wrapper

        assert conv_single.patch_qwen36_causal_conv_single() is True
        composed_wrapper = causal_conv.causal_conv1d_fn_npu
        assert composed_wrapper is not batched_wrapper
        assert composed_wrapper.__wrapped__ is batched_wrapper
        assert getattr(composed_wrapper, conv_batched._PATCH_MARKER)
        assert getattr(composed_wrapper, conv_single._PATCH_MARKER)
        assert backend.causal_conv1d_fn_npu is composed_wrapper
        assert backend.causal_conv1d_fn is composed_wrapper

        # Both installers see their marker on the outer wrapper and do not
        # produce another layer or disturb SGLang's by-value aliases.
        conv_batched.patch_qwen36_causal_conv_prefill()
        assert conv_single.patch_qwen36_causal_conv_single() is True
        assert causal_conv.causal_conv1d_fn_npu is composed_wrapper
        assert composed_wrapper.__wrapped__ is batched_wrapper
        assert backend.causal_conv1d_fn_npu is composed_wrapper
        assert backend.causal_conv1d_fn is composed_wrapper

        b1_inputs = _inputs(1)
        b1_output = _invoke(composed_wrapper, b1_inputs)
        assert torch.equal(b1_output, b1_inputs["x"])
        assert native_calls[-1]["ndim"] == 2
        assert native_calls[-1]["batch_size"] == 1
        assert native_calls[-1]["return_final_states"] is True
        assert not stock_calls

        b2_inputs = _inputs(2)
        b2_output = _invoke(composed_wrapper, b2_inputs)
        assert torch.equal(b2_output, b2_inputs["x"])
        assert native_calls[-1]["ndim"] == 3
        assert native_calls[-1]["batch_size"] == 2
        assert native_calls[-1]["return_final_states"] is True
