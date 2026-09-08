# Copyright 2026 FlagOS Contributors

"""Unit regression tests for the Ascend prefill-tuning plugin patch."""

from __future__ import annotations

import inspect
import os
import sys
import types
import unittest
from unittest import mock

import torch

from sglang_fl.dispatch.backends.vendor.ascend.patches import prefill_tuning


class _UnquantizedLinearMethod:
    def __init__(self):
        self.calls = []


def _linear_apply(self, layer, x, bias):
    self.calls.append((layer, x, bias))
    return "ORIGINAL_LINEAR"


def _packages(leaves):
    """Mock parents too: do not partially import SGLang or register real ops."""
    modules = dict(leaves)
    for name in list(leaves):
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            package = modules.setdefault(parent, types.ModuleType(parent))
            package.__path__ = []
    for name, module in modules.items():
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    return modules


def _nz_modules(disabled=False, fmt=None):
    torch_npu = types.ModuleType("torch_npu")
    if fmt is not None:
        torch_npu.get_npu_format = mock.Mock(return_value=fmt)
        torch_npu.npu_format_cast = mock.Mock()
    unquant = types.ModuleType("sglang.srt.layers.quantization.unquant")
    unquant.UnquantizedLinearMethod = _UnquantizedLinearMethod
    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = types.SimpleNamespace(
        SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=types.SimpleNamespace(
            get=mock.Mock(return_value=disabled)
        )
    )
    return (
        _packages(
            {
                "torch_npu": torch_npu,
                "sglang.srt.environ": environ,
                "sglang.srt.layers.quantization.unquant": unquant,
            }
        ),
        torch_npu,
    )


class _GatingOriginal:
    def __init__(self):
        self.calls = []

    def __call__(self, A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        self.calls.append((A_log, a, b, dt_bias, beta, threshold))
        return "ORIGINAL", "BETA"


class PrefillTuningUnitTest(unittest.TestCase):
    def setUp(self):
        self._old_linear_apply = getattr(_UnquantizedLinearMethod, "apply", None)
        self.old_env = {
            key: os.environ.get(key)
            for key in ("SGLANG_FL_PREFILL_NZ_GATE_UP", "SGLANG_FL_GDN_GATING_BLOCK24")
        }
        os.environ["SGLANG_FL_PREFILL_NZ_GATE_UP"] = "0"
        os.environ["SGLANG_FL_GDN_GATING_BLOCK24"] = "0"

    def tearDown(self):
        if self._old_linear_apply is None:
            if hasattr(_UnquantizedLinearMethod, "apply"):
                delattr(_UnquantizedLinearMethod, "apply")
        else:
            _UnquantizedLinearMethod.apply = self._old_linear_apply
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_both_switches_default_off_do_not_install(self):
        linear = _UnquantizedLinearMethod
        linear.apply = _linear_apply
        prefill_tuning.patch_prefill_tuning()
        self.assertIs(linear.apply, _linear_apply)

    def test_nz_acl_disabled_falls_back(self):
        modules, _ = _nz_modules(disabled=True)
        original = _linear_apply
        _UnquantizedLinearMethod.apply = original
        os.environ["SGLANG_FL_PREFILL_NZ_GATE_UP"] = "1"
        with mock.patch.dict(sys.modules, modules, clear=False):
            self.assertFalse(prefill_tuning._patch_ephemeral_nz())
        self.assertIs(_UnquantizedLinearMethod.apply, original)

    def test_nz_acl_format_fallback(self):
        modules, torch_npu = _nz_modules(fmt=0)
        _UnquantizedLinearMethod.apply = _linear_apply
        os.environ["SGLANG_FL_PREFILL_NZ_GATE_UP"] = "1"
        with mock.patch.dict(sys.modules, modules, clear=False):
            self.assertTrue(prefill_tuning._patch_ephemeral_nz())
            method = _UnquantizedLinearMethod()
            device = types.SimpleNamespace(type="npu")
            x = mock.Mock(
                ndim=2,
                shape=(16384, 5120),
                device=device,
                dtype=torch.bfloat16,
                is_contiguous=mock.Mock(return_value=True),
            )
            weight = mock.Mock(
                shape=(17408, 5120),
                device=device,
                dtype=torch.bfloat16,
                is_contiguous=mock.Mock(return_value=True),
                requires_grad=False,
            )
            result = method.apply(types.SimpleNamespace(weight=weight), x, None)
        self.assertEqual(result, "ORIGINAL_LINEAR")
        torch_npu.npu_format_cast.assert_not_called()
        self.assertEqual(len(method.calls), 1)

    def test_nz_capture_fallback(self):
        modules, torch_npu = _nz_modules(fmt=2)
        _UnquantizedLinearMethod.apply = _linear_apply
        os.environ["SGLANG_FL_PREFILL_NZ_GATE_UP"] = "1"
        with mock.patch.dict(sys.modules, modules, clear=False):
            self.assertTrue(prefill_tuning._patch_ephemeral_nz())
            method = _UnquantizedLinearMethod()
            device = types.SimpleNamespace(type="npu")
            x = mock.Mock(
                ndim=2,
                shape=(16384, 5120),
                device=device,
                dtype=torch.bfloat16,
                is_contiguous=mock.Mock(return_value=True),
            )
            weight = mock.Mock(
                shape=(17408, 5120),
                device=device,
                dtype=torch.bfloat16,
                is_contiguous=mock.Mock(return_value=True),
                requires_grad=False,
            )
            with mock.patch.object(
                prefill_tuning.torch,
                "npu",
                types.SimpleNamespace(
                    is_current_stream_capturing=mock.Mock(return_value=True)
                ),
                create=True,
            ):
                result = method.apply(types.SimpleNamespace(weight=weight), x, None)
        self.assertEqual(result, "ORIGINAL_LINEAR")
        torch_npu.npu_format_cast.assert_not_called()
        self.assertEqual(len(method.calls), 1)

    def test_nz_signature_mismatch_is_not_installed(self):
        def incompatible(self, x):
            return "BAD"

        _UnquantizedLinearMethod.apply = incompatible
        modules, _ = _nz_modules()
        os.environ["SGLANG_FL_PREFILL_NZ_GATE_UP"] = "1"
        with mock.patch.dict(sys.modules, modules, clear=False):
            self.assertFalse(prefill_tuning._patch_ephemeral_nz())
        self.assertIs(_UnquantizedLinearMethod.apply, incompatible)

    def _gating_context(self, kernel, original):
        gating = types.ModuleType("sgl_kernel_npu.fla.fused_gdn_gating")
        gating.fused_gdn_gating_kernel = kernel
        fla = types.ModuleType("sgl_kernel_npu.fla")
        fla.fused_gdn_gating = gating
        backend = types.ModuleType(
            "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend"
        )
        backend.fused_gdn_gating = original
        triton = types.ModuleType("triton")
        triton_runtime = types.ModuleType("triton.runtime")
        triton_runtime.driver = types.SimpleNamespace()
        triton.runtime = triton_runtime
        modules = {
            "triton": triton,
            "triton.runtime": triton_runtime,
            "sgl_kernel_npu.fla": fla,
            "sgl_kernel_npu.fla.fused_gdn_gating": gating,
            "sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend": backend,
        }
        return mock.patch.dict(sys.modules, _packages(modules), clear=False), backend

    def test_gating_abi_incompatible_does_not_install(self):
        def wrong(a, b):
            return None

        kernel = types.SimpleNamespace(fn=wrong)
        original = _GatingOriginal()
        os.environ["SGLANG_FL_GDN_GATING_BLOCK24"] = "1"
        context, backend = self._gating_context(kernel, original)
        with context:
            self.assertFalse(prefill_tuning._patch_gating())
            self.assertIs(backend.fused_gdn_gating, original)

    def test_gating_abi_missing_is_safe_fallback(self):
        kernel = types.SimpleNamespace()
        original = _GatingOriginal()
        os.environ["SGLANG_FL_GDN_GATING_BLOCK24"] = "1"
        context, backend = self._gating_context(kernel, original)
        with context:
            self.assertFalse(prefill_tuning._patch_gating())
            self.assertIs(backend.fused_gdn_gating, original)

    def test_gating_idempotent_and_cpu_shape_fallback(self):
        names = (
            "g",
            "beta_output",
            "A_log",
            "a",
            "b",
            "dt_bias",
            "batch",
            "seq_len",
            "NUM_HEADS",
            "beta",
            "threshold",
            "BLK_HEADS",
        )

        def abi(*args, **kwargs):
            return None

        abi.__signature__ = inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in names
            ]
        )
        kernel = types.SimpleNamespace(fn=abi)
        original = _GatingOriginal()
        os.environ["SGLANG_FL_GDN_GATING_BLOCK24"] = "1"
        context, backend = self._gating_context(kernel, original)
        with context:
            self.assertTrue(prefill_tuning._patch_gating())
            wrapped = backend.fused_gdn_gating
            self.assertTrue(getattr(wrapped, "_sglang_fl_prefill_gating24"))
            self.assertFalse(prefill_tuning._patch_gating())
            self.assertIs(backend.fused_gdn_gating, wrapped)
            args = (
                torch.zeros((24,), dtype=torch.float32),
                torch.zeros((16384, 24), dtype=torch.bfloat16),
                torch.zeros((16384, 24), dtype=torch.bfloat16),
                torch.zeros((24,), dtype=torch.float32),
            )
            self.assertEqual(wrapped(*args), ("ORIGINAL", "BETA"))
            self.assertEqual(len(original.calls), 1)
            self.assertEqual(
                wrapped(*((args[0], args[1][:64], args[2][:64], args[3]))),
                ("ORIGINAL", "BETA"),
            )
            self.assertEqual(len(original.calls), 2)


if __name__ == "__main__":
    unittest.main()
