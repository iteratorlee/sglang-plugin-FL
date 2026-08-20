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

"""Tests for the Ascend adapter to SGLang's current DSA KV-pool contract."""

import sys
import types

import sglang  # noqa: F401 -- initialize the host before importing the plugin platform
from sglang_fl.platform import PlatformFL


def test_npu_dsa_pool_accepts_oot_factory_contract(monkeypatch) -> None:
    received = {}

    class FakeNPUMLATokenToKVPool:
        def __init__(self, **kwargs) -> None:
            received.update(kwargs)

    module_name = "sglang.srt.hardware_backend.npu.memory_pool_npu"
    adapter_name = "sglang_fl.dispatch.backends.vendor.ascend.dsa_kv_pool"
    fake_module = types.ModuleType(module_name)
    fake_module.NPUMLATokenToKVPool = FakeNPUMLATokenToKVPool
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    sys.modules.pop(adapter_name, None)

    platform = PlatformFL.__new__(PlatformFL)
    platform._device_type = "npu"
    try:
        pool_cls = platform.get_dsa_kv_pool_cls()
        pool_cls(
            size=1024,
            page_size=64,
            dtype=None,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=78,
            device="npu",
            kv_cache_dim=576,
            index_head_dim=128,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=78,
        )
    finally:
        # Do not leak an adapter bound to the fake base into later tests.
        sys.modules.pop(adapter_name, None)

    assert issubclass(pool_cls, FakeNPUMLATokenToKVPool)
    assert received == {
        "size": 1024,
        "page_size": 64,
        "dtype": None,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "layer_num": 78,
        "device": "npu",
        "enable_memory_saver": False,
        "start_layer": 0,
        "end_layer": 78,
    }
