# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Defer unused CPU lengths for reviewed MiniMax decode graphs.

The device input copies and all existing completion fences stay in place.
Only an actual, empty graph-dispatch record list permits skipping update().
Detected contract mismatches materialize the old payload. Rebinding a graph,
output, mode or stream during a replay wait rejects that replay.
"""

import __future__
import ast
import functools
import hashlib
import inspect
import logging
from pathlib import Path
import sys
import textwrap
from types import CodeType, FunctionType

logger = logging.getLogger(__name__)
_HELPER = "_sglang_fl_minimax_lazy_cpu_lengths"
_EXPRESSION = (
    "forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)"
)
_SG_SHA256 = {
    "hardware_backend/npu/graph_runner/npu_graph_runner.py": "eac53a5dc7fe1b259bdc48f549e937fc97ad9c0ef17b1bbd12fb83402cb53b56",
    "hardware_backend/npu/graph_runner/npu_cudagraph_backend.py": "4c2a2a4aa41d5076ca2618f259697f5d4e2a18b362d891b43a63afd0445f391d",
    "model_executor/runner/decode_cuda_graph_runner.py": "55892739b9c577ae43a60d5d31eac53f81e2b4aeca57ef5368b9c881117889d8",
    "model_executor/cuda_graph_buffer_registry.py": "f13d41c14a5f4f1484d04d5394d847da059ea721a571693b8d8015b0dedc7ee9",
    "layers/attention/minimax_sparse_backend.py": "2798456cbe4f907fbbccf1eaab1114d2324e79bf1205adb6d4c2e9b8b738b048",
}
_TORCH_NPU_SHA256 = "5ed98b6b5bbbaa162513801e877a24671b58ec2b8eaba3c8d446357efe0e5dc1"
_KERNEL_SHA256 = {
    "attention/gqa_share_sparse_attention.py": "c60257620ddf5dc8f2fc3a5f07a4509ecde308200fafe35c9c02ed166cb4fc9d",
    "indexer/flash_block_score_decode.py": "4f2cd5fccc204a534f03c057b3901f8dc1ce3c5043d3e1e632eb534615715dc0",
}


def _materialize(tensor, padding):
    return tensor.cpu().tolist() + [0] * padding


def _canonical_graph_module(public):
    if public.__module__ not in ("torch_npu.npu.graphs", "torch.npu.graphs"):
        raise RuntimeError("Unknown public TorchNPU graph constructor")
    module = sys.modules.get(public.__module__)
    if module is None or getattr(module, "NPUGraph", None) is not public:
        raise RuntimeError("Conflicting public TorchNPU graph constructor")
    return module


def _reviewed_method(module, class_name, method_name):
    path = Path(module.__file__).resolve()
    method = getattr(getattr(module, class_name), method_name)
    compiled = compile(path.read_text(), str(path), "exec", dont_inherit=True,
                       optimize=sys.flags.optimize)
    class_codes = [code for code in compiled.co_consts
                   if isinstance(code, CodeType) and code.co_name == class_name]
    if len(class_codes) != 1:
        raise RuntimeError("Unknown native graph class")
    methods = [code for code in class_codes[0].co_consts
               if isinstance(code, CodeType) and code.co_name == method_name]
    if (
        len(methods) != 1
        or method.__globals__ is not vars(module)
        or method.__module__ != module.__name__
        or method.__qualname__ != class_name + "." + method_name
        or Path(method.__code__.co_filename).resolve() != path
        or inspect.unwrap(method) is not method
        or method.__code__ != methods[0]
    ):
        raise RuntimeError(f"Unknown native graph method: {class_name}.{method_name}")
    return method


class _Lengths:
    __slots__ = (
        "owner", "runner", "batch", "backend", "key", "graph", "output",
        "tensor", "padding", "raw_bs", "mode", "stream",
        "source_stream_key",
    )

    def __init__(self, owner, runner, batch, key, graph, output):
        self.owner, self.runner, self.batch = owner, runner, batch
        self.backend, self.key = runner.backend, key
        self.graph, self.output = graph, output
        self.mode = graph.graph_dispatch_mode
        self.stream = self.mode.update_stream
        self.tensor = batch.seq_lens
        self.padding = runner.bs - runner.raw_bs
        self.raw_bs = runner.raw_bs
        self.source_stream_key = owner.stream_key(owner.current_stream(self.tensor.device))

    def materialize(self):
        return _materialize(self.tensor, self.padding)


def _transform_execute(original, helper):
    if (
        original.__code__.co_freevars
        or list(inspect.signature(original).parameters)
        != ["self", "forward_batch", "pp_proxy_tensors"]
        or _HELPER in original.__globals__
    ):
        raise RuntimeError("Unknown NPUGraphRunner.execute ABI")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = ast.dump(ast.parse(_EXPRESSION, mode="eval").body)
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "seq_lens"
        and ast.dump(node.value) == expected
    ]
    if len(matches) != 1:
        raise RuntimeError("MiniMax CPU length assignment is not unique")
    matches[0].value = ast.copy_location(
        ast.Call(
            func=ast.Name(id=_HELPER, ctx=ast.Load()),
            args=[ast.Name(id=n, ctx=ast.Load()) for n in ("self", "forward_batch", "graph_key")],
            keywords=[],
        ),
        matches[0].value,
    )
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    namespace = dict(original.__globals__)
    namespace[_HELPER] = helper
    exec(
        compile(tree, original.__code__.co_filename, "exec",
                flags=__future__.annotations.compiler_flag, dont_inherit=True),
        namespace,
    )
    compiled_method = namespace[original.__name__]
    # Preserve late bindings in the native module, including fallback paths.
    # The installer registers the one private helper after validation.
    transformed = FunctionType(
        compiled_method.__code__, original.__globals__, original.__name__,
        original.__defaults__, original.__closure__,
    )
    transformed.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(transformed, original)
    transformed._sglang_fl_minimax_empty_graph = 1
    transformed._sglang_fl_empty_graph_global_name = _HELPER
    transformed._sglang_fl_empty_graph_helper = helper
    return transformed


class _Contract:
    def __init__(self, runner_type, backend_type, graph_type, mode_type,
                 hybrid_type, sparse_type, decode_mode, envs, native_indexer,
                 native_attention, current_stream):
        self.runner_type, self.backend_type = runner_type, backend_type
        self.graph_type, self.mode_type = graph_type, mode_type
        self.hybrid_type, self.sparse_type = hybrid_type, sparse_type
        self.decode_mode, self.envs = decode_mode, envs
        self.native_indexer, self.native_attention = native_indexer, native_attention
        self.current_stream = current_stream
        self.graph_update = graph_type.update
        self.mode_update = mode_type.update_capture_record
        self.graph_update_code = self.graph_update.__code__
        self.mode_update_code = self.mode_update.__code__

    @staticmethod
    def _reject(runner, batch, reason):
        recorded = getattr(runner, "_sglang_fl_empty_update_fallbacks", None)
        if recorded is None:
            recorded = runner._sglang_fl_empty_update_fallbacks = set()
        token = (getattr(batch, "forward_mode", None), reason)
        if token not in recorded:
            recorded.add(token)
            model = getattr(runner, "model_runner", None)
            config = getattr(getattr(model, "model_config", None), "hf_config", None)
            logger.warning(
                "FL_NPU_MINIMAX_EMPTY_FALLBACK rank=%s reason=%s mode=%s "
                "runner=%s backend=%s raw_bs=%s bs=%s arch=%s",
                getattr(getattr(runner, "backend", None), "_device_id", None),
                reason, getattr(batch, "forward_mode", None),
                type(runner).__qualname__, type(getattr(runner, "backend", None)).__qualname__,
                getattr(runner, "raw_bs", None), getattr(runner, "bs", None),
                getattr(config, "architectures", None),
            )
        return None

    def _graph(self, runner, batch, key):
        try:
            args = runner.model_runner.server_args
            if type(runner) is not self.runner_type:
                return self._reject(runner, batch, "runner_type")
            if type(runner.backend) is not self.backend_type:
                return self._reject(runner, batch, "backend_type")
            if batch.forward_mode is not self.decode_mode:
                return self._reject(runner, batch, "batch_forward_mode")
            if runner.capture_forward_mode is not self.decode_mode:
                return self._reject(runner, batch, "capture_forward_mode")
            if runner.captured_req_width != 1:
                return self._reject(runner, batch, "captured_req_width")
            if runner.is_dllm or runner.enable_pdmux:
                return self._reject(runner, batch, "dllm_or_pdmux")
            if runner.model_runner.is_draft_worker:
                return self._reject(runner, batch, "draft_worker")
            if not runner.model_runner.spec_algorithm.is_none():
                return self._reject(runner, batch, "model_spec_algorithm")
            if not args.disable_overlap_schedule:
                return self._reject(runner, batch, "overlap_schedule")
            if args.enable_two_batch_overlap or args.enable_single_batch_overlap:
                return self._reject(runner, batch, "batch_overlap")
            if args.enable_lora or args.speculative_algorithm is not None:
                return self._reject(runner, batch, "lora_or_speculative")
            if args.cuda_graph_backend_decode != "full":
                return self._reject(runner, batch, "graph_backend")
            if args.tp_size != 16 or args.dp_size != 4 or args.ep_size != 1:
                return self._reject(runner, batch, "parallel_sizes")
            if not args.enable_dp_attention:
                return self._reject(runner, batch, "dp_attention")
            if list(runner.capture_bs) != [4, 8, 12, 16]:
                return self._reject(runner, batch, "capture_bs")
            if self.envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
                return self._reject(runner, batch, "plan_stream")
            if self.native_indexer() or self.native_attention():
                return self._reject(runner, batch, "native_sparse_ops")
            if runner.model_runner.model_config.hf_config.architectures != ["MiniMaxM3SparseForConditionalGeneration"]:
                return self._reject(runner, batch, "architecture")
            if batch.seq_lens.device.type != "npu":
                return self._reject(runner, batch, "length_device")
            if batch.seq_lens.device.index != runner.backend._device_id:
                return self._reject(runner, batch, "length_device_index")
            if runner.buffers.seq_lens.device != batch.seq_lens.device:
                return self._reject(runner, batch, "input_buffer_device")
            if not 0 <= runner.raw_bs <= runner.bs:
                return self._reject(runner, batch, "batch_sizes")
            if key.size not in (4, 8, 12, 16) or key.size != runner.bs:
                return self._reject(runner, batch, "shape_size")
            if key.stream_idx is not None or key.variant_label is not None or key.attention_variant is not None:
                return self._reject(runner, batch, "shape_variant")
            attention = runner._replay_attn_backend()
            if type(attention) is not self.hybrid_type:
                return self._reject(runner, batch, "attention_type")
            if type(attention.sparse) is not self.sparse_type:
                return self._reject(runner, batch, "sparse_type")
            if runner._get_update_attr_name() != "actual_seq_lengths_kv":
                return self._reject(runner, batch, "update_attribute")
            if type(runner._get_update_attr_type()) is not list:
                return self._reject(runner, batch, "update_payload_type")
            graph = runner.backend._graphs[key]
            mode = graph.graph_dispatch_mode
            if type(graph) is not self.graph_type:
                return self._reject(runner, batch, "graph_type")
            if graph.auto_dispatch_capture is not True:
                return self._reject(runner, batch, "auto_dispatch_capture")
            if type(graph).update is not self.graph_update or self.graph_update.__code__ is not self.graph_update_code:
                return self._reject(runner, batch, "graph_update_method")
            if getattr(graph.update, "__func__", None) is not self.graph_update or getattr(graph.update, "__self__", None) is not graph:
                return self._reject(runner, batch, "bound_graph_update")
            if type(mode) is not self.mode_type:
                return self._reject(runner, batch, "dispatch_mode_type")
            if type(mode).update_capture_record is not self.mode_update or self.mode_update.__code__ is not self.mode_update_code:
                return self._reject(runner, batch, "dispatch_update_method")
            if getattr(mode.update_capture_record, "__func__", None) is not self.mode_update or getattr(mode.update_capture_record, "__self__", None) is not mode:
                return self._reject(runner, batch, "bound_dispatch_update")
            if type(mode.graph_dispatch_records) is not list or mode.graph_dispatch_records:
                return self._reject(runner, batch, "dispatch_records")
            if not callable(getattr(mode.update_stream, "synchronize", None)):
                return self._reject(runner, batch, "update_stream")
            return graph, runner.backend._outputs[key]
        except (AttributeError, KeyError, TypeError) as error:
            return self._reject(runner, batch, type(error).__name__ + ": " + str(error))

    def lengths(self, runner, batch, key):
        current = self._graph(runner, batch, key)
        if current is None:
            return _materialize(batch.seq_lens, runner.bs - runner.raw_bs)
        return _Lengths(self, runner, batch, key, *current)

    @staticmethod
    def stream_key(stream):
        # current_stream() returns a new Python wrapper for the same native
        # stream. Object identity is not a valid stream comparison.
        return stream.device, stream.stream_id

    def check_stream(self, value, replay_stream=None):
        current = self.stream_key(self.current_stream(value.tensor.device))
        if current != value.source_stream_key or (
            replay_stream is not None
            and self.stream_key(replay_stream) != value.source_stream_key
        ):
            raise RuntimeError("NPU graph stream changed after deferred CPU lengths")

    def resolve(self, value, backend, key, attr_name, attr_type, cpu_update_input):
        if type(value) is not _Lengths or value.owner is not self:
            return False
        if (
            value.backend is not backend or value.runner.backend is not backend
            or value.key is not key
            or attr_name != "actual_seq_lengths_kv" or type(attr_type) is not list
            or cpu_update_input is not None
            or value.runner.raw_bs != value.raw_bs
            or value.runner.bs - value.runner.raw_bs != value.padding
            or value.batch.seq_lens is not value.tensor
        ):
            return False
        current = self._graph(value.runner, value.batch, key)
        return (
            current is not None and current[0] is value.graph
            and current[1] is value.output
            and current[0].graph_dispatch_mode is value.mode
            and value.mode.update_stream is value.stream
        )

    @staticmethod
    def is_lengths(value):
        return type(value) is _Lengths

    @staticmethod
    def diagnostics(value):
        metadata = {
            "seq_lens": None, "seq_lens_min": None, "seq_lens_max": None,
            "seq_lens_source": "not_collected", "empty_graph_update": True,
            "empty_graph_id": hex(id(value.graph)),
        }
        host = getattr(value.batch, "seq_lens_cpu", None)
        if host is not None and host.device.type == "cpu":
            lengths = host[:value.raw_bs].tolist()
            if len(lengths) == value.raw_bs:
                lengths += [0] * value.padding
                metadata.update(
                    seq_lens=lengths, seq_lens_source="host_metadata",
                    seq_lens_min=min(lengths) if lengths else None,
                    seq_lens_max=max(lengths) if lengths else None,
                )
        return metadata


def prepare_empty_graph_update():
    """Validate all sources/ABI before the serial patch installs either hook."""
    import torch
    import torch_npu  # noqa: F401
    import sgl_kernel_npu
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.graph_runner import npu_graph_runner
    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        MiniMaxHybridAttnBackend, MiniMaxSparseAttnBackend,
        _native_attn_enabled, _native_indexer_enabled,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    # TorchNPU aliases torch.npu and torch_npu.npu; importing the graphs
    # submodule through the alias can create a second set of class objects.
    # Validate the module owning the public constructor used by SGLang.
    graph_type = torch.npu.NPUGraph
    graphs = _canonical_graph_module(graph_type)

    root = Path(npu_graph_runner.__file__).resolve().parents[3]
    for name, expected in _SG_SHA256.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unknown SGLang source for empty graph update: {name}")
    if hashlib.sha256(Path(graphs.__file__).read_bytes()).hexdigest() != _TORCH_NPU_SHA256:
        raise RuntimeError("Unknown TorchNPU graph update implementation")
    kernel_root = Path(sgl_kernel_npu.__file__).resolve().parent
    for name, expected in _KERNEL_SHA256.items():
        if hashlib.sha256((kernel_root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unknown MiniMax kernel source: {name}")
    for class_name, method_name in (
        ("NPUGraph", "__init__"), ("NPUGraph", "update"),
        ("_GraphDispatchMode", "update_capture_record")
    ):
        method = _reviewed_method(graphs, class_name, method_name)
        if (
            method.__module__ != graphs.__name__
            or method.__globals__ is not vars(graphs)
            or inspect.unwrap(method) is not method
            or list(inspect.signature(method).parameters) != (
                ["self"] if method_name == "__init__" else ["self", "cpu_update_input"]
            )
        ):
            raise RuntimeError("Conflicting TorchNPU graph update method")
    runner = npu_graph_runner.NPUGraphRunner
    if not graphs.NPUGraph.__weakrefoffset__:
        raise RuntimeError("Empty graph counters require weak-referenceable graphs")
    original = _reviewed_method(npu_graph_runner, "NPUGraphRunner", "execute")
    if original.__module__ != npu_graph_runner.__name__ or inspect.unwrap(original) is not original:
        raise RuntimeError("Conflicting NPUGraphRunner.execute patch")
    contract = _Contract(
        runner, NPUCudaGraphBackend, graphs.NPUGraph, graphs._GraphDispatchMode,
        MiniMaxHybridAttnBackend, MiniMaxSparseAttnBackend, ForwardMode.DECODE,
        envs, _native_indexer_enabled, _native_attn_enabled, torch.npu.current_stream,
    )
    return runner, original, _transform_execute(original, contract.lengths), contract
