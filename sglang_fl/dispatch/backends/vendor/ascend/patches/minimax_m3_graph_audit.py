# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, CPU-only evidence of DP4 decode graph decisions and execution.

Never changes a graph decision, batch, returned value, collective or fence.
Binary records cover every forward; non-graph records retain the same-round
CPU metadata. Unknown evidence fails the external gate, not model execution.
"""

import contextvars
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import sys
import threading
import time
from types import CodeType

logger = logging.getLogger(__name__)
RECORD = struct.Struct("<4sQQQiiiiiiii")
_context = contextvars.ContextVar("fl_minimax_graph_audit", default=None)
_pending = None
_proof_seq = 0
_forward_seq = 0
_writer = None
_installed = False
_started = False
_SOURCES = {
    "sglang.srt.managers.scheduler_components.dp_attn": "5e6b1ce9db4b5bf0d255686226e95caf96398d4050965b72295d53dbd5e965cc",
    "sglang.srt.managers.schedule_batch": "6d9857feb200b5b8fa92517950c54e8e3381e52089d63880cee05e03924bca51",
    "sglang.srt.managers.tp_worker": "3fd8496b1d7b122bfc30edf9f31b9529e96903fc0da26b7217beb29054044f12",
    "sglang.srt.model_executor.forward_batch_info": "a5f7fe7d5211727de3616f5609ed1d2a235bb3beacb0fe1b41a065b48582c7e8",
    "sglang.srt.model_executor.model_runner": "b076adcd47e7543157b4a9a51716b0e3777412d774ec2282489545a2539df0f8",
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner": "55892739b9c577ae43a60d5d31eac53f81e2b4aeca57ef5368b9c881117889d8",
    "sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner": "eac53a5dc7fe1b259bdc48f549e937fc97ad9c0ef17b1bbd12fb83402cb53b56",
}


def classify(state):
    """1 FULL replay, 2 proven DP-mixed eager, 3 prefill, 4 unknown."""
    try:
        assert state["scope_ok"] and state["association_ok"] and state["completed"]
        rows = state["matrix"]
        assert len(rows) == 4 and all(len(row) == 8 for row in rows)
        assert all(all(type(x) is int for x in row) for row in rows)
        assert all(row[0] >= 0 and row[5] in (1, 2, 3, 4) for row in rows)
        assert all(row[2] == int(row[5] in (2, 4)) for row in rows)
        assert all(row[3] == int(row[5] in (1, 3)) for row in rows)
        tokens = tuple(row[0] for row in rows)
        assert tokens == tuple(state["tokens"])
        dp = state["rank"] // 4
        assert rows[dp][5] == state["mode"]
        assert state["vote"] == bool(min(row[2] for row in rows))
        assert state["extend"] == bool(max(row[3] for row in rows))
        mode, actual, eligible = state["mode"], state["actual"], state["eligible"]
        assert state["replays"] == int(actual)
        if mode in (1, 3):
            assert not actual and state["calls"] == 0 and eligible == -1
            return 3
        assert state["calls"] == 1 and eligible in (0, 1)
        assert actual == bool(eligible) and 0 <= state["batch_size"] <= 16
        assert tokens[dp] == (state["batch_size"] if mode == 2 else 0)
        if state["vote"]:
            assert max(tokens) <= 16 and actual
            assert state["bucket"] in (4, 8, 12, 16)
            assert state["bucket"] == next(b for b in (4, 8, 12, 16) if b >= max(tokens))
            return 1
        assert state["extend"] and not actual
        assert any(i != dp and row[5] in (1, 3) for i, row in enumerate(rows))
        return 2
    except (AssertionError, KeyError, TypeError, ValueError, IndexError, StopIteration):
        return 4


def _native(module, qualname):
    code = compile(Path(module.__file__).read_text(), module.__file__, "exec", dont_inherit=True)
    obj = module
    for name in qualname.split("."):
        code = next(x for x in code.co_consts if isinstance(x, CodeType) and x.co_name == name)
        obj = getattr(obj, name)
    # classmethod access exposes __func__; static methods are plain functions.
    obj = getattr(obj, "__func__", obj)
    if (obj.__module__ != module.__name__ or obj.__globals__ is not vars(module)
            or obj.__qualname__ != qualname or obj.__code__ != code):
        raise RuntimeError("Unknown graph-audit binding: " + qualname)
    return obj


def _write_all(fd, data):
    while data:
        count = os.write(fd, data)
        if count <= 0:
            raise OSError("short graph-audit write")
        data = data[count:]


def _record(state):
    global _writer
    directory = Path(os.environ["SGLANG_FL_ASCEND_GRAPH_AUDIT_DIR"])
    rank = state["rank"]
    if _writer is None:
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"rank-{rank:02d}"
        metadata = {
            "version": 1, "record_bytes": RECORD.size, "pid": os.getpid(), "rank": rank,
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_path": __file__,
            "source_sha256": _SOURCES, "format": RECORD.format,
            "activation": "first scheduler CPU gather; initialization before that is outside trace",
        }
        # No append to another process/run: existing identities must fail closed.
        with stem.with_suffix(".meta.json").open("x") as stream:
            json.dump(metadata, stream)
        binary = os.open(stem.with_suffix(".bin"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        details = os.open(stem.with_suffix(".jsonl"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _writer = (rank, os.getpid(), binary, details)
        logger.warning("FL_NPU_GRAPH_AUDIT rank=%s pid=%s bytes=%s", rank, os.getpid(), RECORD.size)
    assert _writer[:2] == (rank, os.getpid())
    category = classify(state)
    if category != 1:
        detail = {k: v for k, v in state.items() if k not in ("fb", "runner")}
        detail["category"] = category
        _write_all(_writer[3], (json.dumps(detail, separators=(",", ":")) + "\n").encode())
    _write_all(_writer[2], RECORD.pack(
        b"FGA1", state["wall_ns"], state["seq"], state.get("proof_seq", 0),
        rank, state["mode"], int(state["actual"]), state["eligible"], category,
        state["batch_size"], max(state.get("tokens", (0,))), state["bucket"],
    ))


def _scope(runner, fb, eager_class, backend_class, graph_class, execute, can_run):
    graph = runner.decode_cuda_graph_runner
    ps = runner.ps
    return (type(graph) is graph_class and type(graph.backend) is backend_class
            and graph_class.execute is execute and graph_class.can_run_graph is can_run
            and getattr(graph.execute, "__func__", None) is execute
            and getattr(graph.execute, "__self__", None) is graph
            and getattr(graph.can_run_graph, "__func__", None) is can_run
            and getattr(graph.can_run_graph, "__self__", None) is graph
            and getattr(backend_class, "_sglang_fl_stage_diagnostics", False)
            and os.environ.get("SGLANG_FL_ASCEND_STAGE_DIAGNOSTICS") == "1"
            and (runner.prefill_cuda_graph_runner is None or type(runner.prefill_cuda_graph_runner) is eager_class)
            and runner.spec_algorithm.is_none() and not runner.enable_elastic_ep
            and runner.lora_manager is None and runner.hisparse_coordinator is None
            and ps.tp_size == 16 and ps.pp_size == 1 and ps.dp_size == 4
            and ps.attn_dp_size == 4 and ps.attn_tp_size == 4
            and ps.attn_dp_rank == ps.tp_rank // 4 and runner.gpu_id == ps.tp_rank
            and graph.require_mlp_tp_gather and graph.require_mlp_sync
            and list(graph.capture_bs) == [4, 8, 12, 16] and graph.max_bs == 16
            and not any((graph.disable_padding, graph.is_encoder_decoder,
                         graph.enable_two_batch_overlap, graph.enable_pdmux, graph.ragged_verify_mode))
            and graph.captured_req_width == 1 and graph.speculative_algorithm is None
            and fb.replace_embeds is None and fb.spec_info is None)


def patch_minimax_m3_graph_audit():
    global _installed
    if os.environ.get("SGLANG_FL_ASCEND_GRAPH_AUDIT") != "1" or _installed:
        return
    try:
        if sys.flags.optimize:
            raise RuntimeError("Graph audit requires Python assertions enabled")
        import importlib
        from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import NPUCudaGraphBackend
        from sglang.srt.model_executor.runner.eager_runner import EagerRunner

        modules = {name: importlib.import_module(name) for name in _SOURCES}
        for name, module in modules.items():
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != _SOURCES[name]:
                raise RuntimeError("Unknown graph-audit source " + name)
        from . import graph_stages

        if hashlib.sha256(Path(graph_stages.__file__).read_bytes()).hexdigest() != "dfba84a199358574fd6ea0250042d1dad9ddb49dd9b62fc77a252f24613fdd80":
            raise RuntimeError("Unknown graph stage evidence")
        dp = modules["sglang.srt.managers.scheduler_components.dp_attn"]
        mr = modules["sglang.srt.model_executor.model_runner"]
        dr = modules["sglang.srt.model_executor.runner.decode_cuda_graph_runner"]
        nr = modules["sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner"]
        original_update = _native(dp, "_update_gather_batch")
        original_forward = _native(mr, "ModelRunner._forward_raw")
        original_can = _native(dr, "DecodeCudaGraphRunner.can_run_graph")
        _native(dr, "DecodeCudaGraphRunner.execute")
        original_execute = _native(nr, "NPUGraphRunner.execute")
        _native(modules["sglang.srt.model_executor.forward_batch_info"], "ForwardBatch.init_new")
        _native(modules["sglang.srt.model_executor.forward_batch_info"], "ForwardBatch.init_mlp_sync_metadata")
        _native(modules["sglang.srt.managers.tp_worker"], "TpModelWorker.forward_batch_generation")
        if not os.environ.get("SGLANG_FL_ASCEND_GRAPH_AUDIT_DIR"):
            raise RuntimeError("Missing unique graph-audit directory")
    except Exception:
        logger.exception("FL_NPU_GRAPH_AUDIT_ERROR installation failed")
        return

    @functools.wraps(original_update)
    def update(batch, info, require_mlp_tp_gather, skip_global_metadata=False):
        global _pending, _proof_seq, _started
        result = original_update(batch, info, require_mlp_tp_gather, skip_global_metadata)
        _started = True
        _pending = None
        try:
            assert require_mlp_tp_gather and not skip_global_metadata
            assert info.tp0_info_cpu.device.type == "cpu"
            matrix = tuple(tuple(row) for row in info.tp0_info_cpu.tolist())
            _proof_seq += 1
            _pending = (batch.global_num_tokens, tuple(batch.global_num_tokens), matrix,
                        threading.get_ident(), _proof_seq)
        except Exception:
            logger.exception("FL_NPU_GRAPH_AUDIT_ERROR metadata unavailable")
        return result

    @functools.wraps(original_can)
    def can_run(self, fb):
        result = original_can(self, fb)
        state = _context.get()
        if state is not None:
            state["calls"] += 1
            if state["fb"] is fb and state["runner"].decode_cuda_graph_runner is self:
                state["eligible"] = int(bool(result))
            else:
                state["association_ok"] = False
        return result

    @functools.wraps(original_forward)
    def forward(self, fb, *args, **kwargs):
        global _pending, _forward_seq
        if not _started:
            return original_forward(self, fb, *args, **kwargs)
        proof, _pending = _pending, None
        _forward_seq += 1
        state = dict(rank=self.ps.tp_rank, mode=int(fb.forward_mode), batch_size=int(fb.batch_size),
                     calls=0, eligible=-1, actual=False, bucket=0, replays=-1, completed=False,
                     seq=_forward_seq, proof_seq=0, wall_ns=time.time_ns(),
                     association_ok=False, scope_ok=False, tokens=(0,), matrix=(),
                     vote=bool(fb.can_run_decode_cuda_graph), extend=bool(fb.is_extend_in_batch),
                     fb=fb, runner=self)
        backend = getattr(self.decode_cuda_graph_runner, "backend", None)
        before = getattr(backend, "_sglang_fl_diagnostic_step", 0)
        try:
            state["scope_ok"] = _scope(self, fb, EagerRunner, NPUCudaGraphBackend,
                                        nr.NPUGraphRunner, original_execute, can_run)
            if proof is not None:
                ref, tokens, matrix, thread, seq = proof
                state.update(tokens=tokens, matrix=matrix, proof_seq=seq)
                state["association_ok"] = (fb.original_global_num_tokens_cpu is ref
                    and tuple(ref) == tokens and thread == threading.get_ident()
                    and _context.get() is None)
        except Exception:
            logger.exception("FL_NPU_GRAPH_AUDIT_ERROR entry observation failed")
        token = _context.set(state)
        try:
            result = original_forward(self, fb, *args, **kwargs)
            try:
                state["completed"] = True
                state["actual"] = bool(result.can_run_graph)
                state["bucket"] = int(getattr(self.decode_cuda_graph_runner, "bs", 0)) if state["actual"] else 0
            except Exception:
                state["scope_ok"] = False
                logger.exception("FL_NPU_GRAPH_AUDIT_ERROR exit observation failed")
            return result
        finally:
            _context.reset(token)
            try:
                state["replays"] = getattr(backend, "_sglang_fl_diagnostic_step", 0) - before
                _record(state)
            except Exception:
                logger.exception("FL_NPU_GRAPH_AUDIT_ERROR record failed")

    dp._update_gather_batch = update
    mr.ModelRunner._forward_raw = forward
    dr.DecodeCudaGraphRunner.can_run_graph = can_run
    _installed = True
    logger.warning("Enabled FL MiniMax same-round graph audit; graph decisions unchanged")
