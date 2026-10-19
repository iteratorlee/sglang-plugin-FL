# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 MiniMax NPU main/index KV store fusion.

Only the native sparse pool store is replaced. Projection, normalization, RoPE,
attention and graph/sampler synchronization remain on their original paths.
"""

import functools
import hashlib
import logging
import os
from pathlib import Path
from types import CodeType

logger = logging.getLogger(__name__)
_NPU_SHA = "876e3962a83f6baae2e55c1370910b3bb24afed085f9b8d76582866595967212"
_POOL_SHA = "7fa13f4cc17f25ed44d09c1b5eaf8a5e5dbd81af3819ab2a2ab49629fd4ee9f6"
_kernel = None
_torch = None
_classes = None
_logged = set()


def _inputs_supported(pool, loc, k, v, index_k, index_v, scales):
    if (_classes is None or type(pool) is not _classes[0]
            or type(pool.main_pool) is not _classes[1]
            or type(pool.index_k_pool) is not _classes[2]
            or index_v is not None
            or any(type(x) not in (type(None), float, int) or x not in (None, 1.0)
                   for x in scales)):
        return False
    main, index = pool.main_pool, pool.index_k_pool
    if (pool.layer_transfer_counter is not None
            or main.layer_transfer_counter is not None
            or index.layer_transfer_counter is not None
            or main.is_quantized_kv_cache):
        return False
    for part in (main, index):
        if (part.dtype != _torch.bfloat16 or part.store_dtype != part.dtype
                or part.use_fia or part.page_size != 128
                or part.head_num != 1 or part.head_dim != 128
                or part.kernel_page_blocks != 1):
            return False
    if main.v_head_dim != 128 or main.size != index.size:
        return False
    tensors = (loc, k, v, index_k)
    if any(type(x) is not _torch.Tensor for x in tensors):
        return False
    if (loc.ndim != 1 or loc.dtype not in (_torch.int32, _torch.int64)
            or loc.device.type != "npu" or not loc.is_contiguous()):
        return False
    rows = loc.numel()
    return 0 < rows <= 262144 and all(x.shape == (rows, 1, 128) and x.dtype == _torch.bfloat16
               and x.device == loc.device and x.is_contiguous()
               for x in (k, v, index_k))


def _buffers_supported(buffers, device):
    first = buffers[0]
    return (type(first) is _torch.Tensor
            and first.ndim == 4 and first.shape[1:] == (128, 1, 128)
            and 0 < first.numel() // 128 < 2**31
            and all(type(x) is _torch.Tensor and x.shape == first.shape
                    and x.device == device and x.dtype == _torch.bfloat16
                    and x.is_contiguous() for x in buffers))


def _make_store(original):
    @functools.wraps(original)
    def store(self, layer, loc, cache_k, cache_v, cache_idx_k, cache_idx_v,
              k_scale=None, v_scale=None, idx_k_scale=None, idx_v_scale=None):
        scales = (k_scale, v_scale, idx_k_scale, idx_v_scale)
        if (_inputs_supported(self, loc, cache_k, cache_v, cache_idx_k, cache_idx_v, scales)
                and type(layer.layer_id) is int
                and self.main_pool.start_layer <= layer.layer_id < self.main_pool.end_layer
                and layer.layer_id in self.index_k_layer_id_mapping):
            # Preserve wrapper AND child-pool transfer waits. No direct access to
            # k_buffer/v_buffer, alternate stream, or additional synchronization.
            k_buffer, v_buffer = self.get_kv_buffer(layer.layer_id)
            index_buffer = self.get_index_k_buffer(layer.layer_id)
            buffers = (k_buffer, v_buffer, index_buffer)
            input_storage = {x.untyped_storage().data_ptr() for x in (cache_k, cache_v, cache_idx_k, loc)}
            if _buffers_supported(buffers, loc.device):
                output_storage = {x.untyped_storage().data_ptr() for x in buffers}
            else:
                output_storage = set()
            if len(output_storage) == 3 and not input_storage.intersection(output_storage):
                rows = loc.numel()
                if rows:
                    _kernel[(min(rows, 48),)](
                        cache_k, cache_v, cache_idx_k, loc,
                        k_buffer, v_buffer, index_buffer, rows,
                        capacity=k_buffer.numel() // 128,
                    )
                    key = (loc.device.index, rows)
                    if key not in _logged:
                        logger.warning("FL_NPU_MINIMAX_KV_STORE device=%s rows=%s loc_dtype=%s", *key, loc.dtype)
                        _logged.add(key)
                return
        # No candidate kernel has been submitted on this path. Preserve every
        # native conversion/scaling/FIA/index-V branch and its exact arguments.
        return original(self, layer, loc, cache_k, cache_v, cache_idx_k, cache_idx_v,
                        k_scale, v_scale, idx_k_scale, idx_v_scale)

    store._sglang_fl_minimax_kv_store = 1
    return store


def _native_method(module, class_name, method_name):
    cls = getattr(module, class_name)
    method = cls.__dict__[method_name]
    code = compile(Path(module.__file__).read_text(), module.__file__, "exec", dont_inherit=True)
    class_code = next(x for x in code.co_consts
                      if isinstance(x, CodeType) and x.co_name == class_name)
    method_code = next(x for x in class_code.co_consts
                       if isinstance(x, CodeType) and x.co_name == method_name)
    if (method.__module__ != module.__name__ or method.__globals__ is not vars(module)
            or method.__qualname__ != class_name + "." + method_name
            or method.__code__ != method_code):
        raise RuntimeError(f"Unknown MiniMax pool method {class_name}.{method_name}")
    return method


def patch_minimax_m3_kv_store():
    global _kernel, _torch, _classes
    if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_FUSED_KV_STORE") != "1":
        return
    try:
        import torch
        from sglang.srt.hardware_backend.npu import memory_pool_npu as npu
        from sglang.srt.mem_cache import memory_pool as pool

        cls = npu.NPUMiniMaxSparseKVPool
        if getattr(cls.set_fused_kv_index_buffer, "_sglang_fl_minimax_kv_store", None) == 1:
            return
        for module, expected in ((npu, _NPU_SHA), (pool, _POOL_SHA)):
            if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Unknown MiniMax NPU pool source")
        original = _native_method(pool, "MiniMaxSparseKVPool", "set_fused_kv_index_buffer")
        if (cls.set_fused_kv_index_buffer is not original
                or cls.__bases__ != (pool.MiniMaxSparseKVPool,)):
            raise RuntimeError("Unknown MiniMax NPU store binding")
        # These are the actual getters/waits and native fallback producers.
        for class_name, target, names in (
            ("MiniMaxSparseKVPool", cls,
             ("get_kv_buffer", "get_index_k_buffer", "_wait_for_layer",
              "set_kv_buffer", "set_index_k_buffer")),
            ("MHATokenToKVPool", npu.NPUMHATokenToKVPool,
             ("get_kv_buffer", "get_key_buffer", "get_value_buffer",
              "_get_key_buffer", "_get_value_buffer")),
            ("MHATokenToKOnlyPool", npu.NPUMHATokenToKOnlyPool, ("get_key_buffer",)),
        ):
            for name in names:
                if getattr(target, name) is not _native_method(pool, class_name, name):
                    raise RuntimeError(f"Unknown resolved MiniMax getter/setter {name}")
        for class_name, names in (("NPUMHATokenToKVPool", ("set_kv_buffer",)),
                                  ("NPUMHATokenToKOnlyPool", ("set_k_buffer", "_get_key_buffer"))):
            for name in names:
                if getattr(getattr(npu, class_name), name) is not _native_method(npu, class_name, name):
                    raise RuntimeError(f"Unknown resolved NPU buffer method {name}")
        from .minimax_m3_kv_store_kernel import store_main_index_kv
    except Exception:
        logger.exception("MiniMax NPU fused KV store disabled")
        return
    _torch = torch
    _classes = (cls, npu.NPUMHATokenToKVPool, npu.NPUMHATokenToKOnlyPool)
    _kernel = store_main_index_kv
    cls.set_fused_kv_index_buffer = _make_store(original)
    logger.warning("Enabled FL MiniMax NPU fused BF16 KV store")
