"""CPU contracts for startup configuration; never import real SG/vendor code."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_fl import native, native_allocator_warmup as policy


@pytest.fixture
def source(tmp_path, monkeypatch):
    # Isolate environment mutations made by the production setdefault call.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop(policy.FLAG, None)
    hashes = {}
    for relative in policy.SOURCE_HASHES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture: " + relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(policy, "SOURCE_HASHES", hashes)
    runner = tmp_path / "srt/model_executor/model_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "class ModelRunner:\n    def init_cuda_graphs(self):\n"
        + textwrap.indent(policy._HOOK_PREFIX.strip(), "        ")
        + "\n"
    )
    monkeypatch.setattr(policy, "_source_root", lambda: tmp_path)
    return tmp_path


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_default_optout_and_proof(source, value):
    if value is not None:
        os.environ[policy.FLAG] = value
    before = dict(os.environ)
    proof = policy.configure()
    assert proof["value"] == (value or "1")
    assert proof["defaulted_in_this_process"] is (value is None)
    assert proof["status"] == ("disabled" if value == "0" else "requested")
    assert dict(os.environ) == {**before, policy.FLAG: value or "1"}
    assert policy.configure()["defaulted_in_this_process"] is False


def test_explicit_optout_needs_no_source(source, monkeypatch):
    os.environ[policy.FLAG] = "0"
    monkeypatch.setattr(policy, "_source_root", Mock(side_effect=AssertionError))
    assert policy.configure()["status"] == "disabled"


@pytest.mark.parametrize("value", ["", "true", "2", "01", " 1"])
def test_invalid_value_never_changes_environment(source, value):
    os.environ[policy.FLAG] = value
    before = dict(os.environ)
    with pytest.raises(ValueError, match=policy.FLAG):
        policy.configure()
    assert dict(os.environ) == before


@pytest.mark.parametrize("changed", ["warmup", "allocator", "hook", "missing"])
def test_unknown_source_rejected_before_defaulting(source, changed):
    if changed in ("warmup", "allocator"):
        relative = list(policy.SOURCE_HASHES)[changed == "allocator"]
        (source / relative).write_text("# changed")
    else:
        path = source / "srt/model_executor/model_runner.py"
        if changed == "missing":
            path.unlink()
        else:
            # A hook after capture is too late, even though both calls exist.
            path.write_text(
                path.read_text().replace("        if (", "        pass\n        if (")
            )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        policy.configure()
    assert policy.FLAG not in os.environ


def test_native_sets_flag_after_identity_before_registration(source, monkeypatch):
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", "0")
    monkeypatch.setenv("SGLANG_FL_SGL_KERNEL_WARMUP_COMPAT", "0")

    def identity():
        assert policy.FLAG not in os.environ
        return {"flagtree": "0.7.0+ascend3.5"}, []

    monkeypatch.setattr(native, "validate_environment", identity)
    from sglang_fl import flagtree_driver, native_registration

    monkeypatch.setattr(
        flagtree_driver, "patch_flagtree_driver_utils", lambda: "not_applicable_version"
    )
    monkeypatch.setitem(sys.modules, "flag_gems", SimpleNamespace())

    def register(*args, **kwargs):
        assert os.environ[policy.FLAG] == "1"
        return SimpleNamespace(registered=[("ones", "PrivateUse1")])

    from sglang_fl import native_prefix_profile

    monkeypatch.setattr(
        native_prefix_profile, "configure", lambda: {"status": "patched"}
    )
    monkeypatch.setattr(native_registration, "enable_ones", register)
    emit = Mock()
    monkeypatch.setattr(native, "_emit", emit)
    native.load_native_plugin()
    assert emit.call_args.kwargs["allocator_warmup"]["status"] == "requested"
    assert emit.call_args.kwargs["flagcx_enabled"] is False


def test_native_identity_failure_cannot_enable_warmup(source, monkeypatch):
    monkeypatch.setattr(
        native, "validate_environment", Mock(side_effect=RuntimeError("identity"))
    )
    with pytest.raises(RuntimeError, match="identity"):
        native.load_native_plugin()
    assert policy.FLAG not in os.environ


@pytest.fixture
def audited_source():
    root = os.environ.get("SGLANG_FL_WARMUP_TEST_SOURCE")
    if not root:
        pytest.skip(
            "set SGLANG_FL_WARMUP_TEST_SOURCE to the audited full SG package directory"
        )
    root = Path(root)
    policy._validate_source(root)
    path = root / next(iter(policy.SOURCE_HASHES))
    spec = importlib.util.spec_from_file_location("allocator_warmup_cpu_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)  # top level imports only stdlib
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_actual_warmup_covers_observed_keys_and_scratch_only(
    audited_source, monkeypatch
):
    warm = audited_source
    # Source-derived reserve=2*max(4*1,5)=10. Real row width is read at startup.
    plan = warm.WarmupPlan(14, 1048649, 64, 10, "npu:0")
    cases = warm.variants(plan)
    for bs_upper in (8, 16):
        for offset in (0, 1):
            assert any(
                c.kernel == "extend"
                and c.bs_upper == bs_upper
                and c.free_offset == offset
                and c.max_extend == 2048
                and c.lens_dtype == "int32"
                for c in cases
            )
    assert len(cases) == 380
    allocated = []

    def allocate(kind, *args, **kwargs):
        assert kwargs.pop("device") == "npu:0"
        tensor = getattr(torch, kind)(*args, **kwargs, device="cpu")
        allocated.append((tensor, tensor.clone()))
        return tensor

    sync = Mock()
    fake_torch = SimpleNamespace(
        int32=torch.int32,
        int64=torch.int64,
        zeros=lambda *a, **k: allocate("zeros", *a, **k),
        full=lambda *a, **k: allocate("full", *a, **k),
        npu=SimpleNamespace(synchronize=sync),
    )
    launches = []

    class Kernel:
        arg_names = (
            "pre_lens_ptr seq_lens_ptr last_loc_ptr free_page_ptr "
            "out_indices bs_upper page_size max_num_extend_tokens BLOCK_SIZE"
        ).split()

        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args):
                if self.name == "extend":
                    start, end, last, free, output, bs, page, extent = args
                    assert torch.count_nonzero(start).item() == 0
                    assert torch.count_nonzero(end).item() == 0
                    assert torch.all(last == -1).item()
                    assert output.dtype == torch.int64
                    assert free.storage_offset() in (0, 1)
                    assert grid[0] <= bs and page == 64 and extent > 0
                else:
                    reqs, row, start, end, output, width, bs = args
                    assert torch.count_nonzero(start).item() == 0
                    assert torch.count_nonzero(end).item() == 0
                    assert row.shape == (1, width) and output.dtype == torch.int32
                launches.append((self.name, grid))
                return object()

            return launch

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel_npu.mem_cache.allocator",
        SimpleNamespace(alloc_extend_kernel=Kernel("extend")),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.allocation",
        SimpleNamespace(assign_req_to_token_pool=Kernel("assign")),
    )
    result = warm.warm_scratch(plan)
    assert result == {
        "launches": 380,
        "unique_specializations": 155,
        "scratch_row_bytes": 4194596,
    }
    assert len(launches) == 380
    sync.assert_called_once_with()
    for tensor, before in allocated:
        assert torch.equal(tensor, before)
    # The launch doubles validate zero-work inputs and scratch ownership;
    # actual vendor numerical/graph execution remains an NPU acceptance gate.
