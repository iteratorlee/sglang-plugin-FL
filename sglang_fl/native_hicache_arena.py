"""Audited P-only GLM53 KV/Mamba sharing of an explicitly sized host arena.

No SG imports at plugin load. The existing early opt-in owns scope/validation;
only its allocation request and subsequent host buffer ownership are adapted.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import logging
import math
import os
from pathlib import Path
import sys
from types import FunctionType
import weakref

FLAG = "SGLANG_GLM53_HICACHE_EARLY_HOST_ALLOC"
PREFIX = "sglang.srt.mem_cache.pool_host."
# SG a6ba7fb1b75cd79a34191f2b6ca29aa694f85167 (R5). Full-file, fail-closed.
SOURCE_HASHES = {
    "srt/mem_cache/pool_host/glm53.py": "14cbecd76bd1e2237f13b3be8c8f590317b41c5dcebb1141bb9a0ac1c48bcc0c",
    "srt/mem_cache/pool_host/mamba.py": "da4ed97df42ff47ccceb9cf5302575a870e59e611e63f99e0f7b2073c7710763",
    "srt/mem_cache/pool_host/base.py": "e10f8806b88693670fbd7373f3aa4d11e11deb550e5548e64b9a96c56aaefbaf",
    "srt/mem_cache/pool_host/common.py": "cebdb7ac44fe9980954862722bec84b318ba859027347a8d814dfbd1d9b936d4",
    "srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py": "f6f651b07e21f38f398743e50e2409ecc5944347cb60291b76b29cb7413be9e8",
    "srt/model_executor/model_runner.py": "7ab45c817b614c79e602d838abd19da1fed392099789aadf2026f562727590f2",
}
logger = logging.getLogger(__name__)
_finder = None


def _validate_source(root):
    for relative, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unaudited hybrid arena source: {relative}: {actual}")
    return dict(SOURCE_HASHES)


def _next_power2(size):
    if size <= 0:
        raise ValueError("Hybrid arena budget must be positive")
    return 1 << (size - 1).bit_length()


def _check_arena(arena, *, pinned=True):
    import torch

    if (arena.device.type != "cpu" or arena.dtype != torch.uint8
            or arena.ndim != 1 or not arena.is_contiguous()
            or arena.storage_offset() != 0
            or arena.numel() > arena.untyped_storage().nbytes()):
        raise RuntimeError("Hybrid arena requires a contiguous CPU byte owner")
    if pinned and not arena.is_pinned():
        raise RuntimeError("Hybrid arena owner is not pinned")


def _typed_views(arena, offset, specs, *, pinned=True):
    """Plan all bounds first; narrow only within the tensor's logical extent.

    pinned=False is a CPU-test seam, never used by the runtime patch.
    Zero-sized components still share storage and retain their original dtype.
    """
    _check_arena(arena, pinned=pinned)
    if not 0 <= offset <= arena.numel():
        raise RuntimeError("Hybrid arena KV extent is out of bounds")
    plan = []
    for shape, dtype in specs:
        shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("Negative hybrid buffer dimension")
        alignment = math.lcm(64, dtype.itemsize)
        start = (offset + alignment - 1) // alignment * alignment
        size = math.prod(shape) * dtype.itemsize
        if start + size > arena.numel():
            raise RuntimeError(
                f"Hybrid arena exhausted: end={start + size}, logical={arena.numel()}"
            )
        plan.append((start, size, shape, dtype))
        offset = start + size
    views = []
    storage = arena.untyped_storage()
    for start, size, shape, dtype in plan:
        view = arena.narrow(0, start, size).view(dtype).view(shape)
        if (not view.is_contiguous() or view.untyped_storage()._cdata != storage._cdata
                or view.storage_offset() * view.element_size() != start
                or view.numel() * view.element_size() != size
                or (pinned and size and not view.is_pinned())):
            raise RuntimeError("Hybrid arena typed view failed storage/pinning checks")
        views.append(view)
    return views, offset


class _State:
    def __init__(self):
        self.pending = None  # weak KV pool reference; never owns pinned memory

    def pool(self):
        return self.pending() if self.pending is not None else None


def _install_glm(module, state):
    if getattr(module, "_fl_hybrid_arena", False):
        return
    if module._early_host_arena is not None:
        raise RuntimeError("Hybrid arena must be installed before early reservation")
    reserve = module._reserve_early_host_arena
    cls = module.GLM53MLAPoolHost
    initialize, destroy = cls.__init__, cls.destroy
    claim = module._EarlyHostArena.claim

    @functools.wraps(reserve)
    def reserved(runner):
        if state.pool() is not None:
            raise RuntimeError("Previous hybrid arena still awaits Mamba")
        budget = None

        def allocate(dims, **kwargs):
            nonlocal budget
            budget = dims[0]
            arena = module.ALLOC_MEMORY_FUNCS["npu"]((_next_power2(budget),), **kwargs)
            _check_arena(arena)
            if arena.numel() != _next_power2(budget):
                raise RuntimeError("Hybrid arena allocator returned an unexpected extent")
            return arena

        # Run the exact audited SG validation/allocation code with a private
        # allocator binding. Never replace torch.empty or the shared allocator
        # dictionary. Publish its one global output only on success.
        namespace = dict(reserve.__globals__, ALLOC_MEMORY_FUNCS={"npu": allocate})
        adapted = FunctionType(reserve.__code__, namespace, reserve.__name__,
                               reserve.__defaults__, reserve.__closure__)
        record = adapted(runner)
        record._fl_logical_budget = budget
        module._early_host_arena = record
        logger.info("GLM53 hybrid arena reserved: budget_bytes=%d logical_bytes=%d npu=%d",
                    budget, record.arena.numel(), record.gpu_id)
        return record

    @functools.wraps(claim)
    def claimed(self, size_bytes):
        if size_bytes > getattr(self, "_fl_logical_budget", 0):
            self.discard()
            raise RuntimeError("KV claim exceeds unchanged logical HiCache budget")
        return claim(self, size_bytes)

    @functools.wraps(initialize)
    def initialized(self, *args, **kwargs):
        record = module._early_host_arena
        if record is None:
            return initialize(self, *args, **kwargs)
        # SG commit discards record.arena inside initialize. This local owns the
        # FULL logical tensor throughout commit, not just the narrowed KV view.
        arena = record.arena
        try:
            initialize(self, *args, **kwargs)
            if not record.consumed or arena is None:
                raise RuntimeError("Hybrid KV reservation was not committed")
            self._fl_hybrid_owner = (arena, self._packed_host_arena.numel(), record.gpu_id)
            state.pending = weakref.ref(self)
        except BaseException:
            record.discard()
            self._fl_hybrid_owner = None
            raise

    @functools.wraps(destroy)
    def destroyed(self):
        try:
            return destroy(self)
        finally:
            self._fl_hybrid_owner = None
            if state.pool() is self:
                state.pending = None

    module._reserve_early_host_arena = reserved
    module._EarlyHostArena.claim = claimed
    cls.__init__, cls.destroy = initialized, destroyed
    module._fl_hybrid_arena = True


def _install_mamba(module, state):
    if getattr(module, "_fl_hybrid_arena", False):
        return
    cls = module.MambaPoolHost
    initialize, init_buffer = cls.__init__, cls.init_kv_buffer

    @functools.wraps(initialize)
    def initialized(self, *args, **kwargs):
        pool = state.pool()
        self._fl_hybrid_owner = getattr(pool, "_fl_hybrid_owner", None)
        try:
            initialize(self, *args, **kwargs)
        except BaseException:
            if self._fl_hybrid_owner is not None:
                self.kv_buffer = self.temporal_buffer = None
                self.conv_buffer = []
                self.temporal_staging_buffer = None
                self.conv_staging_buffers = []
                pool.destroy()
            raise
        finally:
            self._fl_hybrid_owner = None
            if pool is not None:
                pool._fl_hybrid_owner = None
                state.pending = None

    @functools.wraps(init_buffer)
    def buffer(self):
        import torch

        owner = self._fl_hybrid_owner
        if owner is None:
            # Never turn a missed early handoff into another late NPU pin.
            if os.environ.get(FLAG) == "1" and str(self.device_pool.device).split(":")[0] == "npu":
                from sglang.srt.runtime_context import get_disagg
                if get_disagg().disaggregation_mode == "prefill":
                    raise RuntimeError("Missing hybrid KV owner before Mamba allocation")
            return init_buffer(self)
        arena, offset, gpu_id = owner
        device = str(self.device_pool.device).split(":")
        if (device[0] != "npu" or (len(device) > 1 and int(device[1]) != gpu_id)
                or torch.device(self.device).type != "cpu" or not self.pin_memory
                or torch.get_device_module("npu").current_device() != gpu_id
                or self.layout not in ("page_first", "page_first_direct")
                or self.temporal_dtype != torch.float32):
            raise RuntimeError("Hybrid Mamba CPU/NPU/layout/FP32 contract mismatch")
        cache = self.device_pool.mamba_cache
        for tensor in (cache.temporal, *cache.conv):
            if tensor.device.type != "npu" or tensor.device.index != gpu_id:
                raise RuntimeError("Hybrid Mamba device tensor belongs to another NPU")
        if cache.temporal.dtype != self.temporal_dtype or any(
            tensor.dtype != self.conv_dtype for tensor in cache.conv
        ):
            raise RuntimeError("Hybrid Mamba dtype contract mismatch")
        prefix = (self.size, self.num_mamba_layers, 1)
        specs = [(prefix + tuple(self.temporal_state_shape), self.temporal_dtype)]
        specs.extend((prefix + tuple(shape), self.conv_dtype) for shape in self.conv_state_shapes)
        views, end = _typed_views(arena, offset, specs)
        self.temporal_buffer, self.conv_buffer = views[0], views[1:]
        logger.info("GLM53 hybrid Mamba arena reused: kv_bytes=%d end_bytes=%d logical_bytes=%d pinned=True",
                    offset, end, arena.numel())
        return [view for view in views if view.numel() > 0]

    cls.__init__, cls.init_kv_buffer = initialized, buffer
    module._fl_hybrid_arena = True


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, finder, name):
        self.wrapped, self.finder, self.name = wrapped, finder, name

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module):
        # Recheck all audited dependencies before executing cache module code.
        _validate_source(self.finder.root)
        self.wrapped.exec_module(module)
        self.finder.install(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root
        self.state = _State()
        self.targets = {PREFIX + "glm53": _install_glm, PREFIX + "mamba": _install_mamba}
        self.done = set()

    def install(self, module):
        self.targets[module.__name__](module, self.state)
        self.done.add(module.__name__)
        if self.done == set(self.targets) and self in sys.meta_path:
            sys.meta_path.remove(self)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        expected = self.root / (fullname.removeprefix("sglang.").replace(".", "/") + ".py")
        if (spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
                or Path(spec.origin).resolve() != expected.resolve()):
            raise RuntimeError(f"Unsupported hybrid arena module loader: {fullname}")
        spec.loader = _Loader(spec.loader, self, fullname)
        return spec


def configure():
    global _finder
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be exactly 0 or 1")
    if value == "0":
        return {"status": "disabled", "env": FLAG}
    from sglang_fl.native_allocator_warmup import _source_root

    root = _source_root()
    hashes = _validate_source(root)
    if _finder is None:
        finder = _Finder(root)
        # Early plugin load normally has neither target loaded. Already loaded
        # modules are supported only before reservation; never import them here.
        loaded = [sys.modules[name] for name in finder.targets if name in sys.modules]
        for module in loaded:
            if (Path(module.__file__).resolve() != root / (module.__name__.removeprefix("sglang.").replace(".", "/") + ".py")
                    or getattr(module.__spec__, "_initializing", False)
                    or getattr(module, "_early_host_arena", None) is not None):
                raise RuntimeError("Hybrid arena installation requires untouched complete modules")
        for module in loaded:
            finder.install(module)
        if finder.done != set(finder.targets):
            sys.meta_path.insert(0, finder)
        _finder = finder
    elif _finder.root != root:
        raise RuntimeError("Hybrid arena SG source root changed")
    return {"status": "deferred" if _finder in sys.meta_path else "installed",
            "env": FLAG, "sglang_source": str(root), "source_sha256": hashes}
