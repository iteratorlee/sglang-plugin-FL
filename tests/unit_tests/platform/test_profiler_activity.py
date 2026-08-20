# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for PlatformFL profiler activity selection."""

import sys
from types import SimpleNamespace

import sglang  # noqa: F401 -- initialize the host before importing its platform plugin
import torch

from sglang_fl.platform import PlatformFL


def _platform_for(device_type: str) -> PlatformFL:
    platform = PlatformFL.__new__(PlatformFL)
    platform._device_type = device_type
    return platform


def test_profiler_activity_names() -> None:
    assert _platform_for("npu").get_torch_profiler_activity_str() == "NPU"
    assert _platform_for("xpu").get_torch_profiler_activity_str() == "XPU"
    assert _platform_for("cuda").get_torch_profiler_activity_str() == "GPU"
    assert _platform_for("musa").get_torch_profiler_activity_str() == "GPU"


def test_npu_profiler_activity(monkeypatch) -> None:
    expected = object()
    fake_torch_npu = SimpleNamespace(
        profiler=SimpleNamespace(ProfilerActivity=SimpleNamespace(NPU=expected))
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    assert _platform_for("npu").get_torch_profiler_activity() is expected


def test_gpu_profiler_activity() -> None:
    assert (
        _platform_for("cuda").get_torch_profiler_activity()
        is torch.profiler.ProfilerActivity.CUDA
    )
