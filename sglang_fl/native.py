"""Opt-in FlagOS integration preserving SGLang's native model implementation.

Only FlagGems ``ones`` and dense TP/attention-TP collectives are replaced.
No platform, model, quantization, FLA, cache, MoE, or PD-transfer patches.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import socket
from collections import Counter
from importlib import metadata
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
GROUPS = frozenset(("tp", "attention_tp"))


def _emit(event, **fields):
    record = dict(event=event, pid=os.getpid(), host=socket.gethostname(), **fields)
    line = json.dumps(record, sort_keys=True)
    logger.warning("SGLANG_FL_NATIVE %s", line)
    directory = os.environ.get("SGLANG_FL_PROOF_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"{record['host']}-{record['pid']}.jsonl").open("a") as stream:
            stream.write(line + "\n")


def validate_environment():
    """Fail before installing any hook or operator if the scope is ambiguous."""
    expected = {
        "USE_FLAGGEMS": "1",
        "SGLANG_FL_FLAGOS_WHITELIST": "ones",
        "SGLANG_FL_FLAGOS_BLACKLIST": "",
        "SGLANG_FL_OOT_ENABLED": "0",
        "SGLANG_FL_DISAGG_FLAGCX": "0",
        "SGLANG_FL_DIST_BACKEND": "hccl",
        "SGLANG_PLATFORM": "",
        "SGLANG_PLUGINS": "sglang_fl",
    }
    for key, default in expected.items():
        value = os.environ.get(key, default).strip()
        if value != default:
            raise ValueError(
                f"native integration requires {key}={default!r}, got {value!r}"
            )
    if os.environ.get("SGLANG_FL_SGL_KERNEL_WARMUP_COMPAT", "0") not in ("0", "1"):
        raise ValueError("SGLANG_FL_SGL_KERNEL_WARMUP_COMPAT must be 0 or 1")
    from sglang_fl.native_identity import validate_flagtree

    versions = validate_flagtree()
    versions["flag-gems"] = metadata.version("flag-gems")
    root = Path(os.environ.get("FLAGCX_PATH", ""))
    if not os.environ.get("FLAGCX_PATH") or not root.is_absolute():
        raise ValueError(
            "Set FLAGCX_PATH to the absolute FlagCX source/build directory"
        )
    files = [
        root / "build/lib/libflagcx.so",
        root / "plugin/interservice/flagcx_wrapper.py",
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return versions, files


def _eligible(group):
    return (
        group.world_size > 1
        and group.device.type == "npu"
        and group.unique_name.split(":", 1)[0] in GROUPS
    )


def _transport(group, tensor):
    if not _eligible(group) or tensor.device.type != "npu":
        return None
    comm = getattr(group, "_fl_native_comm", None)
    if comm is None or comm.disabled or not comm.available:
        raise RuntimeError(
            f"Required FlagCX communicator unavailable: {group.unique_name}"
        )
    if tensor.device != comm.device or not tensor.is_contiguous():
        raise ValueError("FlagCX requires a contiguous tensor on the group's NPU")
    return comm


def _check_output(output, input, count):
    if (
        output.device != input.device
        or output.dtype != input.dtype
        or not output.is_contiguous()
        or output.numel() != count
    ):
        raise ValueError("FlagCX output device/dtype/contiguity/size mismatch")


def _observer(group):
    counts = Counter()

    def observe(api, tensor):
        counts[api] += 1
        count = counts[api]
        # Log powers of two. This is API submission, not a stream sync or graph
        # replay counter. The observer is attached AFTER communicator warmup.
        if count & (count - 1) == 0:
            _emit(
                "collective_submitted",
                api=api,
                count=count,
                group=group.unique_name,
                ranks=list(group.ranks),
                rank=group.rank_in_group,
                world_size=group.world_size,
                device=str(tensor.device),
                dtype=str(tensor.dtype),
                numel=tensor.numel(),
                phase="sglang_collective",
            )

    return observe


def install_collectives(cls, factory=None):
    """Wrap the existing class in place; preserve aliases and non-scoped calls.

    Validate every target before changing any method. Unlike SG's optional hook
    loader, this installation propagates incompatibility instead of swallowing it.
    ``factory`` is a dependency seam for CPU contract tests.
    """
    if getattr(cls, "_fl_native_installed", False):
        return
    signatures = {
        "all_reduce": ("self", "input_"),
        "_all_gather_into_tensor": ("self", "output", "input"),
        "_reduce_scatter_tensor": ("self", "output", "input"),
        "all_gather": ("self", "input_", "dim", "output_tensor_list"),
        "destroy": ("self",),
    }
    originals = {name: getattr(cls, name) for name in (*signatures, "__init__")}
    for name, parameters in signatures.items():
        actual = tuple(inspect.signature(originals[name]).parameters)
        if actual != parameters:
            raise RuntimeError(f"Unsupported SG {name} signature: {actual}")
    if factory is None:
        from sglang_fl.distributed.device_communicators.flagcx import FlagCXCommunicator

        factory = FlagCXCommunicator

    def initialize(self, *args, **kwargs):
        originals["__init__"](self, *args, **kwargs)
        self._fl_native_comm = None
        if not _eligible(self):
            return
        comm = factory(group=self.cpu_group, device=self.device)
        if comm.disabled or not comm.available:
            raise RuntimeError(f"FlagCX initialization failed for {self.unique_name}")
        self._fl_native_comm = comm
        comm._collective_observer = _observer(self)
        # Some native paths keep a reference to NpuCommunicator and call it
        # directly. Bind only this TP group's instance, leaving its quantized
        # method and every other group's communicator unchanged.
        npu = self.npu_communicator
        if npu is None or npu.disabled:
            raise RuntimeError("Scoped TP group has no enabled native NpuCommunicator")
        native_reduce, native_gather = npu.all_reduce, npu.all_gather

        @functools.wraps(native_reduce)
        def npu_reduce(x):
            if x.device.type != "npu":
                return native_reduce(x)
            return all_reduce(self, x)

        @functools.wraps(native_gather)
        def npu_gather(x, dim=-1):
            if x.device.type != "npu":
                return native_gather(x, dim)
            return all_gather(self, x, dim)

        npu.all_reduce, npu.all_gather = npu_reduce, npu_gather
        _emit(
            "communicator_ready",
            group=self.unique_name,
            ranks=list(self.ranks),
            rank=self.rank_in_group,
            world_size=self.world_size,
            device=str(self.device),
            bootstrap="native_cpu_group",
        )

    def all_reduce(self, input_):
        comm = _transport(self, input_)
        if comm is None:
            return originals["all_reduce"](self, input_)
        return comm.all_reduce(input_)

    def all_gather_into_tensor(self, output, input):
        comm = _transport(self, input)
        if comm is None:
            return originals["_all_gather_into_tensor"](self, output, input)
        _check_output(output, input, input.numel() * self.world_size)
        comm.all_gather(output, input)

    def reduce_scatter_tensor(self, output, input):
        comm = _transport(self, input)
        if comm is None:
            return originals["_reduce_scatter_tensor"](self, output, input)
        if input.numel() % self.world_size:
            raise ValueError("FlagCX reduce-scatter requires equal splits")
        _check_output(output, input, input.numel() // self.world_size)
        comm.reduce_scatter(output, input)
        return output

    def all_gather(self, input_, dim=-1, output_tensor_list=None):
        # List output is deliberately native (including its None return value).
        if output_tensor_list is not None:
            return originals["all_gather"](self, input_, dim, output_tensor_list)
        comm = _transport(self, input_)
        if comm is None:
            return originals["all_gather"](self, input_, dim, output_tensor_list)
        if not -input_.dim() <= dim < input_.dim():
            raise ValueError(f"Invalid all_gather dimension: {dim}")
        dim %= input_.dim()
        shape = tuple(input_.shape)
        output = torch.empty(
            (shape[0] * self.world_size,) + shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        comm.all_gather(output, input_)
        return (
            output.reshape((self.world_size,) + shape)
            .movedim(0, dim)
            .reshape(shape[:dim] + (shape[dim] * self.world_size,) + shape[dim + 1 :])
        )

    def destroy(self):
        comm = getattr(self, "_fl_native_comm", None)
        if comm is not None:
            # Explicit, ordered teardown before SG destroys the bootstrap group.
            comm.close()
            self._fl_native_comm = None
        return originals["destroy"](self)

    replacements = {
        "__init__": initialize,
        "all_reduce": all_reduce,
        "_all_gather_into_tensor": all_gather_into_tensor,
        "_reduce_scatter_tensor": reduce_scatter_tensor,
        "all_gather": all_gather,
        "destroy": destroy,
    }
    for name, replacement in replacements.items():
        setattr(cls, name, functools.wraps(originals[name])(replacement))
    cls._fl_native_installed = True


def load_native_plugin():
    versions, files = validate_environment()
    # Invoke the same narrow helper as legacy mode without importing dispatch
    # or vendor patches. The pinned 0.7.0 driver already caches utils eagerly.
    from sglang_fl.flagtree_driver import patch_flagtree_driver_utils

    driver_utils_cache = patch_flagtree_driver_utils()
    if os.environ.get("SGLANG_FL_SGL_KERNEL_WARMUP_COMPAT", "0") == "1":
        from sglang_fl.native_warmup import install

        install()
        _emit("warmup_compat_ready", target="sgl_kernel_npu.norm.add_rmsnorm_bias")
    import flag_gems
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang_fl.native_registration import enable_ones

    # No YAML dispatch policy, vendor patch import, or broad enable() fallback.
    registration = enable_ones(
        flag_gems,
        record=os.environ.get("SGLANG_FLAGGEMS_RECORD", "0") == "1",
        once=os.environ.get("SGLANG_FLAGGEMS_LOG_ONCE", "1") == "1",
        **(
            {"path": os.environ["SGLANG_FLAGGEMS_LOG_PATH"]}
            if os.environ.get("SGLANG_FLAGGEMS_LOG_PATH")
            else {}
        ),
    )
    install_collectives(GroupCoordinator)
    _emit(
        "plugin_ready",
        integration="native",
        versions=versions,
        driver_utils_cache=driver_utils_cache,
        aten_allowlist=["ones"],
        aten_registrations=registration.registered,
        groups=sorted(GROUPS),
        files={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    )
