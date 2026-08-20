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

"""Ascend decode graph runner compatibility extensions.

SGLang filters decode capture sizes to attention-TP multiples when an MoE A2A
backend requires scheduler-side TP gather.  GLM-5.2 still needs that gather for
eager prefill, but users may want separate logical decode graph buckets below
the TP size.  This adapter records those logical buckets with TP-aligned static
inputs and selects them using the pre-padding request count at replay time.

The behavior is opt-in through ``SGLANG_FL_EXACT_DECODE_GRAPH=1``.  Without the
environment variable this class delegates to SGLang's native NPUGraphRunner.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
    NPUGraphRunner,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx
from sglang.srt.utils import get_cuda_graph_batch_size_alignment
from sglang_fl.dispatch.backends.vendor.ascend.graph_utils import (
    align_capture_size,
    exact_decode_graph_enabled,
    resolve_logical_batch_size,
)

logger = logging.getLogger(__name__)


class AscendNPUGraphRunner(NPUGraphRunner):
    """NPUGraphRunner that can preserve sub-TP logical capture buckets."""

    def __init__(self, *args, **kwargs):
        self._fl_exact_decode_graph = exact_decode_graph_enabled()
        self._fl_logical_capture_size: Optional[int] = None
        self._fl_logical_replay_size: Optional[int] = None
        self._fl_attention_graph_metadata: dict[
            tuple[int, Optional[int]], object
        ] = {}
        super().__init__(*args, **kwargs)

    def capture(self) -> None:
        if self._fl_exact_decode_graph:
            requested = sorted(
                {
                    int(bs)
                    for bs in self.model_runner.server_args.cuda_graph_config.decode.bs
                    if 0 < int(bs) <= self.max_bs
                }
            )
            if not requested:
                raise ValueError(
                    "SGLANG_FL_EXACT_DECODE_GRAPH requires at least one positive "
                    "decode graph batch size within the runner capacity"
                )
            self.capture_bs = requested
            logger.info(
                "Ascend exact logical decode graph capture enabled: bs=%s, "
                "attention-TP alignment=%d",
                requested,
                get_cuda_graph_batch_size_alignment(self.model_runner.server_args),
            )
        super().capture()

    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        if not self._fl_exact_decode_graph or not self.require_gathered_buffer:
            return super().capture_one_shape(
                size, forward, stream_idx=stream_idx, variant_label=variant_label
            )

        alignment = get_cuda_graph_batch_size_alignment(
            self.model_runner.server_args
        )
        internal_size = align_capture_size(size, alignment)
        self._fl_logical_capture_size = (
            size if internal_size != size else None
        )
        try:
            result = super().capture_one_shape(
                internal_size,
                forward,
                stream_idx=stream_idx,
                variant_label=variant_label,
            )
            attn_backend = self._attention_backend_for_stream(stream_idx)
            self._fl_attention_graph_metadata[(size, stream_idx)] = (
                attn_backend.graph_metadata[internal_size]
            )
            return result
        finally:
            self._fl_logical_capture_size = None

    def _attention_backend_for_stream(self, stream_idx: Optional[int]):
        if self.enable_pdmux:
            if stream_idx is None:
                raise RuntimeError("PDMux graph metadata requires a stream index")
            return self.model_runner.decode_attn_backend_group[stream_idx]
        return self.attn_backend

    def _capture_graph_size(self, *, bs: int, num_tokens: int) -> int:
        if self._fl_logical_capture_size is not None:
            return self._fl_logical_capture_size
        if self._fl_logical_replay_size is not None:
            return self._fl_logical_replay_size
        return super()._capture_graph_size(bs=bs, num_tokens=num_tokens)

    def _make_graph_key(
        self,
        size: int,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        if self._fl_logical_replay_size is not None:
            size = self._fl_logical_replay_size
        return super()._make_graph_key(size, stream_idx, variant_label)

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if not self._fl_exact_decode_graph or not self.require_gathered_buffer:
            return super().load_batch(forward_batch, pp_proxy_tensors)

        logical_raw_bs = resolve_logical_batch_size(
            getattr(forward_batch, "_original_batch_size", None),
            forward_batch.batch_size,
        )
        logical_bs = self._pad_to_bucket(int(logical_raw_bs), self.capture_bs)
        alignment = get_cuda_graph_batch_size_alignment(
            self.model_runner.server_args
        )
        internal_capture_bs = sorted(
            {align_capture_size(size, alignment) for size in self.capture_bs}
        )
        internal_bs = align_capture_size(logical_bs, alignment)
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        attn_backend = self._attention_backend_for_stream(stream_idx)
        metadata_key = (logical_bs, stream_idx)
        if metadata_key not in self._fl_attention_graph_metadata:
            raise RuntimeError(
                "Missing retained attention metadata for logical decode graph "
                f"{logical_bs} on stream {stream_idx}"
            )
        # Multiple logical graphs below TP size share the same internal shape.
        # Retain and restore each graph's own metadata tensors so their captured
        # device pointers remain live and the replay updater touches the right
        # storage.
        attn_backend.graph_metadata[internal_bs] = (
            self._fl_attention_graph_metadata[metadata_key]
        )

        original_capture_bs = self.capture_bs
        self._fl_logical_replay_size = logical_bs
        try:
            # Base replay preparation must see the communication-aligned size:
            # it selects attention metadata and fills all static input buffers.
            # _make_graph_key still maps the replay to the distinct logical graph.
            self.capture_bs = internal_capture_bs
            result = super().load_batch(forward_batch, pp_proxy_tensors)
        finally:
            self.capture_bs = original_capture_bs

        logger.debug(
            "Ascend exact decode graph replay selected logical_bs=%d "
            "for TP-aligned internal_bs=%d",
            logical_bs,
            self.bs,
        )
        return result

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        self._fl_logical_replay_size = None
        try:
            return super().execute(forward_batch, pp_proxy_tensors)
        finally:
            self._fl_logical_replay_size = None
