"""Finite native profile extension checked against the real frozen policy."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from sglang_fl import native_prefix_profile as patch


@pytest.fixture
def policy():
    source = os.environ.get("SGLANG_FL_PREFIX_TEST_SOURCE")
    if not source:
        pytest.skip("set SGLANG_FL_PREFIX_TEST_SOURCE to the audited policy file")
    name = "native_prefix_profile_cpu_contract"
    spec = importlib.util.spec_from_file_location(name, Path(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def config(tp=8, dp=2, ep=8, nnodes=1):
    return SimpleNamespace(
        device="npu",
        disaggregation_mode="decode",
        disaggregation_transfer_backend="ascend",
        quantization="modelslim",
        speculative_draft_model_quantization="modelslim",
        tp_size=tp,
        dp_size=dp,
        ep_size=ep,
        pp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        nnodes=nnodes,
        moe_dp_size=1,
        dwdp_size=1,
        enable_prefill_cp=False,
        enable_dp_attention=True,
        page_size=64,
        enable_hierarchical_cache=False,
        disaggregation_decode_retraction_backup=None,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=4,
        speculative_num_draft_tokens=5,
    )


def model(arch="Glm5NextForConditionalGeneration"):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))


def test_topology_extension_preserves_alias_and_native_code(policy, monkeypatch):
    monkeypatch.setenv("SGLANG_GLM53_2NODE_PD", "1")
    fn = policy.glm53_pd_decode_prefix_profile
    before = fn.__code__
    assert fn(config(), model()) is None
    assert patch.install(policy) == "patched"
    assert policy.glm53_pd_decode_prefix_profile is fn
    assert fn(config(), model()) == policy.Glm53PDDecodePrefixProfile()
    assert patch.install(policy) == "already_supported"
    assert before.co_code == fn.__code__.co_code
    assert before.co_names == fn.__code__.co_names
    assert sum(a != b for a, b in zip(before.co_consts, fn.__code__.co_consts)) == 1
    for topology in [(16, 8, 16), (8, 4, 8)]:
        assert fn(config(*topology), model()) is not None
    assert fn(config(16, 8, 16, nnodes=2), model()) is not None
    assert fn(config(8, 2, 8, nnodes=2), model()) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "cuda"),
        ("disaggregation_mode", "prefill"),
        ("disaggregation_transfer_backend", "mooncake"),
        ("quantization", "bf16"),
        ("speculative_draft_model_quantization", None),
        ("tp_size", 4),
        ("dp_size", 1),
        ("ep_size", 4),
        ("pp_size", 2),
        ("dcp_size", 2),
        ("attn_cp_size", 2),
        ("enable_dp_attention", False),
        ("page_size", 128),
        ("enable_hierarchical_cache", True),
        ("disaggregation_decode_retraction_backup", "other"),
        ("speculative_algorithm", "NONE"),
        ("speculative_eagle_topk", 2),
        ("speculative_num_steps", 2),
        ("speculative_num_draft_tokens", 6),
    ],
)
def test_other_guards_still_reject(policy, key, value):
    patch.install(policy)
    cfg = config()
    setattr(cfg, key, value)
    assert policy.glm53_pd_decode_prefix_profile(cfg, model()) is None


def test_other_architecture_rejected(policy):
    patch.install(policy)
    assert policy.glm53_pd_decode_prefix_profile(config(), model("OtherModel")) is None


def test_unknown_gate_and_callable_fail_without_mutation():
    def fn(cfg, model_config):
        return (cfg, model_config)

    module = SimpleNamespace(glm53_pd_decode_prefix_profile=fn)
    before = fn.__code__
    with pytest.raises(RuntimeError, match="Unknown GLM prefix topology"):
        patch.install(module)
    assert fn.__code__ is before
    with pytest.raises(RuntimeError, match="callable"):
        patch.install(SimpleNamespace(glm53_pd_decode_prefix_profile=Mock()))


def test_configure_disable_and_invalid(monkeypatch):
    monkeypatch.setenv(patch.FLAG, "0")
    monkeypatch.setattr(
        patch.importlib, "import_module", Mock(side_effect=AssertionError)
    )
    assert patch.configure()["status"] == "disabled"
    for value in ["", "true", "2"]:
        monkeypatch.setenv(patch.FLAG, value)
        with pytest.raises(ValueError):
            patch.configure()
