# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Routing/transaction tests; actual cache copying needs the native device gate."""

import builtins
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / (
    "sglang_fl/dispatch/backends/vendor/ascend/patches/minimax_m3_kv_store.py")


def load():
    spec = importlib.util.spec_from_file_location("kv_store_test_target", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    next_storage = 100

    def __init__(self, shape, dtype="bf16"):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.device = SimpleNamespace(type="npu", index=0)
        self.contiguous = True
        Tensor.next_storage += 1
        self.storage = Tensor.next_storage

    def numel(self):
        result = 1
        for item in self.shape:
            result *= item
        return result

    def is_contiguous(self):
        return self.contiguous

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage)


class Main:
    dtype = store_dtype = "bf16"
    use_fia = False
    page_size = head_dim = v_head_dim = 128
    head_num = kernel_page_blocks = 1
    size = 256
    start_layer, end_layer = 4, 7
    layer_transfer_counter = None
    is_quantized_kv_cache = False


class Index(Main):
    pass


class Sparse:
    layer_transfer_counter = None

    def __init__(self):
        self.main_pool, self.index_k_pool = Main(), Index()
        self.index_k_layer_id_mapping = {5: 0, 6: 1}
        self.buffers = [Tensor((3, 128, 1, 128)) for _ in range(3)]
        self.events = []

    def get_kv_buffer(self, layer):
        self.events.append(("main_get", layer))
        return self.buffers[:2]

    def get_index_k_buffer(self, layer):
        self.events.append(("index_get", layer))
        return self.buffers[2]


class Routing(unittest.TestCase):
    def setUp(self):
        self.module = load()
        self.module._classes = (Sparse, Main, Index)
        self.module._torch = SimpleNamespace(Tensor=Tensor, bfloat16="bf16", int32="i32", int64="i64")
        self.launches, self.fallbacks = [], []
        self.pool = Sparse()
        self.inputs = [Tensor((4,), "i64")] + [Tensor((4, 1, 128)) for _ in range(3)] + [None]
        self.layer = SimpleNamespace(layer_id=6)
        launches = self.launches

        class Kernel:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    launches.append((grid, args, kwargs))
                return run

        self.module._kernel = Kernel()

        def native(*args):
            self.fallbacks.append(args)
            return "native_return"

        self.store = self.module._make_store(native)

    def call(self, *scales):
        return self.store(self.pool, self.layer, *self.inputs, *scales)

    def test_waits_before_one_launch(self):
        self.assertIsNone(self.call())
        self.assertEqual(self.pool.events, [("main_get", 6), ("index_get", 6)])
        self.assertEqual(len(self.launches), 1)
        self.assertFalse(self.fallbacks)
        self.assertEqual(self.launches[0][0], (4,))
        self.assertEqual(self.launches[0][2], {"capacity": 384})

    def test_unsupported_inputs_keep_original_arguments(self):
        for option in ("index_v", "noncontiguous", "dtype", "transfer", "layer"):
            with self.subTest(option=option):
                self.setUp()
                if option == "index_v":
                    self.inputs[-1] = Tensor((4, 1, 128))
                elif option == "noncontiguous":
                    self.inputs[1].contiguous = False
                elif option == "dtype":
                    self.inputs[1].dtype = "fp16"
                elif option == "transfer":
                    self.pool.layer_transfer_counter = object()
                else:
                    self.layer.layer_id = 4
                self.assertEqual(self.call(), "native_return")
                self.assertEqual(self.fallbacks[0][2:7], tuple(self.inputs))
                self.assertFalse(self.launches)

    def test_nonunit_scale_and_alias_fall_back(self):
        self.assertEqual(self.call(2.0), "native_return")
        self.assertFalse(self.launches)
        self.pool.buffers[0].storage = self.inputs[1].storage
        self.assertEqual(self.call(), "native_return")
        self.assertFalse(self.launches)

    def test_launch_exception_never_retries_native(self):
        class Broken:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    raise RuntimeError("partial submission")
                return run
        self.module._kernel = Broken()
        with self.assertRaisesRegex(RuntimeError, "partial submission"):
            self.call()
        self.assertFalse(self.fallbacks)

    def test_disabled_does_not_import_framework(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "sglang", "triton"):
                raise AssertionError("framework import while disabled")
            return original(name, *args, **kwargs)

        with patch.dict(os.environ, {"SGLANG_FL_ASCEND_MINIMAX_FUSED_KV_STORE": "0"}), patch(
                "builtins.__import__", guarded):
            self.module.patch_minimax_m3_kv_store()


if __name__ == "__main__":
    unittest.main()
