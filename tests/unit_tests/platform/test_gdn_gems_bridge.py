"""CPU wiring tests; actual tensor semantics are verified by FlagGems pytest."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def load(relative):
    path = Path(__file__).resolve().parents[3] / relative
    spec = importlib.util.spec_from_file_location("gdn_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bridge_preserves_all_arguments_and_uses_dispatch(monkeypatch):
    dispatch = ModuleType("sglang_fl.dispatch")
    calls = []
    dispatch.call_op = lambda name, **kw: calls.append((name, kw)) or "output"
    monkeypatch.setitem(sys.modules, "sglang_fl.dispatch", dispatch)
    bridge = load("sglang_fl/dispatch/bridge/fla_sigmoid_gating.py")
    args = list(range(11))
    assert (
        bridge.fused_sigmoid_gating_delta_rule_update_bridge(*args, is_kda=True)
        == "output"
    )
    assert calls[0][0] == "fused_sigmoid_gating_delta_rule_update"
    assert calls[0][1]["initial_state_source"] == 9
    assert calls[0][1]["is_kda"] is True


def test_backend_uses_gems_export_without_native_fallback(monkeypatch):
    gems = ModuleType("flag_gems.runtime.backend._ascend.fla")
    calls = []
    gems.fused_sigmoid_gating_delta_rule_update = (
        lambda *a, **kw: calls.append((a, kw)) or "gems"
    )
    monkeypatch.setitem(sys.modules, gems.__name__, gems)
    backend = load("sglang_fl/dispatch/backends/vendor/ascend/impl/gdn_gems.py")
    assert backend.fused_sigmoid_gating_delta_rule_update_gems(1, scale=0.5) == "gems"
    assert calls == [((1,), {"scale": 0.5})]


def test_enabled_patch_routes_model_to_bridge_and_missing_gems_fails(monkeypatch):
    def module(name):
        if name in sys.modules:
            return sys.modules[name]
        value = ModuleType(name)
        value.__path__ = []
        monkeypatch.setitem(sys.modules, name, value)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(module(parent), child, value)
        return value

    for name in list(sys.modules):
        if name in ("sglang", "sglang_fl", "flag_gems") or name.startswith(
            ("sglang.", "sglang_fl.", "flag_gems.")
        ):
            monkeypatch.delitem(sys.modules, name)
    chunk = module("sglang.srt.layers.attention.fla.chunk")
    recurrent = module("sglang.srt.layers.attention.fla.fused_recurrent")
    gdn = module("sglang.srt.layers.attention.linear.kernels.gdn_triton")
    chunk.chunk_gated_delta_rule = lambda: None
    recurrent.fused_recurrent_gated_delta_rule = lambda: None
    recurrent.fused_recurrent_gated_delta_rule_packed_decode = lambda: None
    original = lambda: "native"
    gdn.fused_sigmoid_gating_delta_rule_update = original
    bridge = lambda: "Gems bridge"
    for name, attribute in [
        ("fla_chunk", "chunk_gated_delta_rule_bridge"),
        ("fla_fused_recurrent", "fused_recurrent_gated_delta_rule_bridge"),
        ("fla_packed_decode", "fused_recurrent_gated_delta_rule_packed_decode_bridge"),
        ("fla_sigmoid_gating", "fused_sigmoid_gating_delta_rule_update_bridge"),
    ]:
        setattr(module("sglang_fl.dispatch.bridge." + name), attribute, bridge)
    gems = module("flag_gems.runtime.backend._ascend.fla")
    gems.fused_sigmoid_gating_delta_rule_update = lambda: "Gems"
    monkeypatch.setenv("SGLANG_FL_GEMS_GDN_DECODE", "1")
    patch = load("sglang_fl/dispatch/fla_patch.py")
    patch.patch_fla_functions()
    assert gdn.fused_sigmoid_gating_delta_rule_update is bridge
    gems.fused_sigmoid_gating_delta_rule_update = None
    import pytest

    with pytest.raises(RuntimeError, match="FlagGems"):
        patch.patch_fla_functions()
