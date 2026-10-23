"""Serialize Ascend graph updates and replay when explicitly requested.

Author: liyan <yan.l@pku.edu.cn>
"""

import functools
import logging
import os
import weakref

logger = logging.getLogger(__name__)


def patch_ascend_graph_updates():
    """Keep task updates off the replay stream and propagate update failures."""
    if os.environ.get("SGLANG_FL_ASCEND_SERIAL_GRAPH_UPDATE") != "1":
        return

    import numpy as np
    import torch
    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )

    if getattr(NPUCudaGraphBackend, "_sglang_fl_serial_graph_update", False):
        return
    original = NPUCudaGraphBackend.replay_with_input_update
    diagnostics = os.environ.get("SGLANG_FL_ASCEND_STAGE_DIAGNOSTICS") == "1"
    empty_patch = None
    if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_EMPTY_GRAPH_UPDATE") == "1":
        try:
            if not diagnostics:
                raise RuntimeError("Empty graph update requires stage diagnostics")
            from .minimax_m3_empty_graph import prepare_empty_graph_update

            empty_patch = prepare_empty_graph_update()
        except Exception:
            # Keep all existing vendor patches available on an unsupported
            # source. The external cold/worker gate rejects a missing marker.
            logger.exception("MiniMax empty graph update disabled")
    empty_contract = empty_patch[3] if empty_patch is not None else None

    def stage(phase, **metadata):
        if diagnostics:
            from .graph_stages import _stage

            _stage(phase, **metadata)

    @functools.wraps(original)
    def replay(
        self,
        shape_key,
        seq_lens,
        attr_name=None,
        attr_type=None,
        cpu_update_input=None,
    ):
        skip_update = False
        lazy_lengths = None
        if empty_contract is not None and empty_contract.is_lengths(seq_lens):
            lazy_lengths = seq_lens
            empty_contract.check_stream(lazy_lengths)
            skip_update = empty_contract.resolve(
                seq_lens, self, shape_key, attr_name, attr_type, cpu_update_input
            )
            if not skip_update:
                seq_lens = seq_lens.materialize()
        if cpu_update_input is None and not skip_update:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        graph = self._graphs[shape_key]
        mode = getattr(graph, "graph_dispatch_mode", None)
        update_stream = getattr(mode, "update_stream", None)
        if update_stream is None or not callable(
            getattr(update_stream, "synchronize", None)
        ):
            raise RuntimeError(
                "Serial NPU graph updates require the TorchNPU graph dispatch "
                "update_stream interface (verified on torch_npu 2.10.0)."
            )
        if diagnostics:
            recorded = getattr(self, "_sglang_fl_recorded_shapes", set())
            if shape_key not in recorded:
                records = mode.graph_dispatch_records
                logger.warning(
                    "FL_NPU_GRAPH_RECORDS rank=%s shape=%s count=%s ops=%s",
                    self._device_id,
                    shape_key,
                    len(records),
                    [str(record.op_cache_entry) for record in records],
                )
                recorded.add(shape_key)
                self._sglang_fl_recorded_shapes = recorded
        metadata = {}
        if diagnostics:
            if skip_update:
                metadata = empty_contract.diagnostics(lazy_lengths)
            else:
                if empty_contract is not None:
                    metadata = {
                        "seq_lens": None, "seq_lens_min": None, "seq_lens_max": None,
                        "empty_graph_id": None,
                    }
                lengths = seq_lens
                if isinstance(lengths, torch.Tensor):
                    lengths = lengths.tolist() if lengths.device.type == "cpu" else None
                if isinstance(lengths, (list, tuple)) and lengths:
                    metadata.update({
                        "seq_lens": list(lengths),
                        "seq_lens_min": min(lengths),
                        "seq_lens_max": max(lengths),
                    })
                if empty_contract is not None:
                    metadata.update(empty_graph_update=False, seq_lens_source="device_payload")
        with torch.npu.device(self._device_id):
            # Retire the preceding replay before updating its recorded tasks.
            replay_stream = torch.npu.current_stream()
            if skip_update:
                empty_contract.check_stream(lazy_lengths, replay_stream)
            stage("graph_previous_wait", **metadata)
            replay_stream.synchronize()
            if skip_update:
                empty_contract.check_stream(lazy_lengths, replay_stream)
            if skip_update and (
                self._graphs.get(shape_key) is not graph
                or self._outputs.get(shape_key) is not lazy_lengths.output
                or graph.graph_dispatch_mode is not mode
                or mode.update_stream is not update_stream
            ):
                raise RuntimeError("NPU graph bindings changed during serial replay wait")
            if skip_update and not empty_contract.resolve(
                lazy_lengths, self, shape_key, attr_name, attr_type, cpu_update_input
            ):
                # The wait may expose a changed graph/record/method. Restore
                # the original payload and update behavior for this call.
                seq_lens = lazy_lengths.materialize()
                cpu_update_input = [{attr_name: seq_lens}]
                skip_update = False
                stage(
                    "graph_update_fallback", empty_graph_update=False,
                    seq_lens=seq_lens,
                    seq_lens_min=min(seq_lens) if seq_lens else None,
                    seq_lens_max=max(seq_lens) if seq_lens else None,
                    seq_lens_source="device_payload",
                    empty_graph_id=None,
                )
            if skip_update:
                counts = getattr(self, "_sglang_fl_empty_update_counts", None)
                if counts is None:
                    counts = self._sglang_fl_empty_update_counts = {}
                previous = counts.get(shape_key)
                count = previous[1] + 1 if previous and previous[0]() is graph else 1
                counts[shape_key] = (weakref.ref(graph), count)
                if count == 1:
                    logger.warning(
                        "FL_NPU_MINIMAX_EMPTY_UPDATE rank=%s shape=%s graph=%s records=0 fences=preserved",
                        self._device_id, shape_key, hex(id(graph)),
                    )
            stage("graph_update_call")
            if not skip_update:
                graph.update(cpu_update_input=cpu_update_input)
            stage("graph_update_wait", update_stream=str(update_stream))
            update_stream.synchronize()
            stage("graph_replay_submit")
            graph.replay()
            stage("graph_wait")
            replay_stream.synchronize()
            stage("graph_done")
        return self._outputs[shape_key]

    NPUCudaGraphBackend.replay_with_input_update = replay
    NPUCudaGraphBackend._sglang_fl_serial_graph_update = True
    if empty_patch is not None:
        runner, _, execute, _ = empty_patch
        execute.__globals__[execute._sglang_fl_empty_graph_global_name] = (
            execute._sglang_fl_empty_graph_helper
        )
        runner.execute = execute
        NPUCudaGraphBackend._sglang_fl_minimax_empty_graph = 1
        logger.warning("Enabled FL guarded MiniMax empty graph update")
    logger.warning("Enabled FL serial NPU graph update and replay completion")
