"""CPU-only routing/ABI contracts; no claim of real NPU execution."""

import ast
import builtins
import ctypes
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_fl
from sglang_fl import flagtree_driver
from sglang_fl import native
from sglang_fl import native_allocator_warmup
from sglang_fl import native_identity
from sglang_fl import native_registration
from sglang_fl.distributed.device_communicators import flagcx


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    # The source guard has its own real-source CPU contracts. These routing
    # tests intentionally use the seven-file SG integration snapshots.
    monkeypatch.setattr(
        native_allocator_warmup, "configure", Mock(return_value={"status": "requested"})
    )
    for key in tuple(os.environ):
        if (
            key.startswith(("SGLANG_FL_", "SGLANG_FLAGGEMS_"))
            and key != "SGLANG_FL_SG_SOURCE"
        ) or key in (
            "SGLANG_PLATFORM",
            "SGLANG_PLUGINS",
            "USE_FLAGGEMS",
            "FLAGCX_PATH",
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sglang_fl, "_plugin_loaded", False)
    monkeypatch.setattr(sglang_fl, "_plugin_active", False)
    monkeypatch.setattr(native_registration, "_libraries", [])


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_FL_INTEGRATION", "native")
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    for name in ("build/lib/libflagcx.so", "plugin/interservice/flagcx_wrapper.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("contract fixture")
    monkeypatch.setattr(
        native.metadata,
        "version",
        lambda name: "test-gems",
    )
    monkeypatch.setattr(
        native_identity,
        "validate_flagtree",
        lambda: {
            "flagtree": "0.7.0+ascend3.5",
            "triton_module": "3.5.1",
        },
    )
    return tmp_path


@pytest.mark.parametrize(
    "key,value",
    [
        ("SGLANG_FL_FLAGOS_WHITELIST", "ones,matmul"),
        ("SGLANG_FL_FLAGOS_WHITELIST", ""),
        ("SGLANG_FL_FLAGOS_BLACKLIST", "add"),
        ("USE_FLAGGEMS", "0"),
        ("SGLANG_PLATFORM", "sglang_fl"),
        ("SGLANG_PLUGINS", "other"),
        ("SGLANG_FL_OOT_ENABLED", "1"),
        ("SGLANG_FL_DIST_BACKEND", "flagcx"),
        ("SGLANG_FL_DISAGG_FLAGCX", "1"),
    ],
)
@pytest.mark.parametrize("flagcx_enabled", ["0", "1"])
def test_reject_scope_expansion(environment, monkeypatch, key, value, flagcx_enabled):
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", flagcx_enabled)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        native.validate_environment()


def test_versions_and_build_artifacts(environment, monkeypatch):
    assert native.validate_environment()[0]["flagtree"] == "0.7.0+ascend3.5"
    monkeypatch.setattr(
        native_identity,
        "validate_flagtree",
        Mock(side_effect=RuntimeError("identity rejected")),
    )
    with pytest.raises(RuntimeError, match="identity rejected"):
        native.validate_environment()


def test_torch_plugin_import_is_not_a_capi_build(environment):
    (environment / "build/lib/libflagcx.so").unlink()
    with pytest.raises(FileNotFoundError):
        native.validate_environment()


@pytest.mark.parametrize("flagcx_enabled", ["0", "1"])
def test_native_entrypoint_does_not_activate_platform_or_full_layers(
    monkeypatch, flagcx_enabled
):
    monkeypatch.setenv("SGLANG_FL_INTEGRATION", "native")
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", flagcx_enabled)
    setup = Mock()
    monkeypatch.setattr(native, "load_native_plugin", setup)
    for name in (
        "_build_config",
        "_init_dispatch",
        "_apply_vendor_patches",
        "_setup_communicator_hooks",
        "_setup_flaggems",
    ):
        monkeypatch.setattr(sglang_fl, name, Mock(side_effect=AssertionError(name)))
    assert sglang_fl.activate_platform() is None
    sglang_fl.load_plugin()
    sglang_fl.load_plugin()
    setup.assert_called_once_with()
    assert sglang_fl.is_plugin_active()


def test_failure_escapes_sg_exception_handler_and_can_retry(monkeypatch):
    monkeypatch.setenv("SGLANG_FL_INTEGRATION", "native")
    setup = Mock(side_effect=[RuntimeError("broken ABI"), None])
    monkeypatch.setattr(native, "load_native_plugin", setup)
    with pytest.raises(SystemExit, match="broken ABI"):
        try:
            sglang_fl.load_plugin()
        except Exception:
            pytest.fail("SG would swallow this failure")
    assert not sglang_fl.is_plugin_active()
    sglang_fl.load_plugin()
    assert sglang_fl.is_plugin_active()


@pytest.mark.parametrize("flagcx_enabled", [None, "0", "1"])
def test_only_ones_and_no_yaml_policy(environment, monkeypatch, flagcx_enabled):
    if flagcx_enabled is not None:
        monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", flagcx_enabled)
    if flagcx_enabled == "0":
        monkeypatch.delenv("FLAGCX_PATH")
        (environment / "build/lib/libflagcx.so").unlink()
        (environment / "plugin/interservice/flagcx_wrapper.py").unlink()
    monkeypatch.setattr(flagtree_driver, "version", lambda name: "0.7.0+ascend3.5")
    cache_hook = Mock(wraps=flagtree_driver.patch_flagtree_driver_utils)
    monkeypatch.setattr(flagtree_driver, "patch_flagtree_driver_utils", cache_hook)
    library = Mock()
    constructor = Mock(return_value=library)
    monkeypatch.setattr(native_registration.torch.library, "Library", constructor)
    fn = Mock()
    gems = SimpleNamespace(
        only_enable=Mock(
            side_effect=lambda lib, **kw: lib.impl("ones", fn, "PrivateUse1")
        ),
        enable=Mock(side_effect=AssertionError),
    )
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.parallel_state",
        SimpleNamespace(GroupCoordinator=object),
    )
    install, emit = Mock(), Mock()
    monkeypatch.setattr(native, "install_collectives", install)
    monkeypatch.setattr(native, "_emit", emit)
    monkeypatch.setenv("SGLANG_FL_CONFIG", "/must/not/read.yaml")
    original_import = builtins.__import__

    def scoped_import(name, *args, **kwargs):
        # Exercise the real entrypoint and cache helper. Neither may activate
        # legacy patches or initialize a driver for the pinned 0.7 wheel.
        assert not name.startswith(
            (
                "sglang_fl.dispatch",
                "sglang_fl.platform",
                "sglang.srt.models",
                "sglang.srt.layers",
                "triton.backends",
            )
        ), name
        if flagcx_enabled == "0":
            assert not name.startswith(
                (
                    "sglang.srt.distributed",
                    "sglang_fl.distributed",
                    "plugin.interservice",
                    "flagcx",
                )
            ), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", scoped_import)
    assert sglang_fl.activate_platform() is None
    sglang_fl.load_plugin()
    sglang_fl.load_plugin()
    assert sglang_fl.is_plugin_active()
    cache_hook.assert_called_once_with()
    if flagcx_enabled == "0":
        install.assert_not_called()
    else:
        install.assert_called_once_with(object)
    emit.assert_called_once()
    assert emit.call_args.args == ("plugin_ready",)
    assert emit.call_args.kwargs["driver_utils_cache"] == "not_applicable_version"
    assert emit.call_args.kwargs["aten_allowlist"] == ["ones"]
    assert emit.call_args.kwargs["flagcx_enabled"] is (flagcx_enabled != "0")
    assert emit.call_args.kwargs["groups"] == (
        [] if flagcx_enabled == "0" else ["attention_tp", "tp"]
    )
    if flagcx_enabled == "0":
        assert emit.call_args.kwargs["files"] == {}
    audited = native_registration._libraries[0]
    constructor.assert_called_once_with("aten", "IMPL")
    gems.only_enable.assert_called_once_with(
        lib=audited, include=["ones"], record=False, once=True
    )
    library.impl.assert_called_once_with("ones", fn, "PrivateUse1")
    gems.enable.assert_not_called()


@pytest.mark.parametrize("flagcx_enabled", ["0", "1"])
def test_cache_hook_cannot_bypass_native_identity_gate(
    environment, monkeypatch, flagcx_enabled
):
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", flagcx_enabled)
    monkeypatch.setattr(
        native_identity,
        "validate_flagtree",
        Mock(side_effect=RuntimeError("identity rejected")),
    )
    cache_hook = Mock()
    monkeypatch.setattr(flagtree_driver, "patch_flagtree_driver_utils", cache_hook)
    with pytest.raises(SystemExit, match="identity rejected"):
        sglang_fl.load_plugin()
    cache_hook.assert_not_called()
    assert not sglang_fl.is_plugin_active()


@pytest.mark.parametrize("flagcx_enabled", ["0", "1"])
def test_swallowed_gems_registration_failure_cannot_report_plugin_ready(
    environment, monkeypatch, flagcx_enabled
):
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", flagcx_enabled)
    library = Mock()
    library.impl.side_effect = RuntimeError("dispatcher rejected registration")
    monkeypatch.setattr(
        native_registration.torch.library, "Library", lambda *a: library
    )

    def only_enable(lib, **kwargs):
        # Match Gems' failure mode: bookkeeping happens first, errors are caught.
        gems.all_keys.append("ones")
        try:
            lib.impl("ones", Mock(), "PrivateUse1")
        except RuntimeError:
            pass

    gems = SimpleNamespace(only_enable=only_enable, all_keys=[])
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.parallel_state",
        SimpleNamespace(GroupCoordinator=object),
    )
    install, emit = Mock(), Mock()
    monkeypatch.setattr(native, "install_collectives", install)
    monkeypatch.setattr(native, "_emit", emit)
    with pytest.raises(SystemExit, match="ones registration incomplete"):
        sglang_fl.load_plugin()
    assert gems.all_keys == ["ones"]
    assert not sglang_fl.is_plugin_active()
    install.assert_not_called()
    emit.assert_not_called()


@pytest.mark.parametrize("value", ["", "2", "true", "false", "01", " 0 "])
def test_flagcx_switch_is_strict(environment, monkeypatch, value):
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", value)
    with pytest.raises(SystemExit, match="SGLANG_FL_FLAGCX_ENABLED"):
        sglang_fl.load_plugin()
    assert not sglang_fl.is_plugin_active()


@pytest.mark.parametrize("path", [None, "relative/not-installed", "/not-installed"])
def test_flagcx_disabled_needs_no_path_or_artifact_checks(
    environment, monkeypatch, path
):
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", "0")
    if path is None:
        monkeypatch.delenv("FLAGCX_PATH")
    else:
        monkeypatch.setenv("FLAGCX_PATH", path)
    monkeypatch.setattr(native, "Path", Mock(side_effect=AssertionError("FlagCX path")))
    version = Mock(return_value="test-gems")
    monkeypatch.setattr(native.metadata, "version", version)
    versions, files = native.validate_environment()
    version.assert_called_once_with("flag-gems")
    assert files == []
    assert versions["flagtree"] == "0.7.0+ascend3.5"
    assert versions["flag-gems"] == "test-gems"


def test_flagcx_disabled_preserves_warmup_shim(environment, monkeypatch):
    monkeypatch.setenv("SGLANG_FL_FLAGCX_ENABLED", "0")
    monkeypatch.setenv("SGLANG_FL_SGL_KERNEL_WARMUP_COMPAT", "1")
    warmup = Mock()
    monkeypatch.setitem(
        sys.modules, "sglang_fl.native_warmup", SimpleNamespace(install=warmup)
    )
    library = Mock()
    monkeypatch.setattr(
        native_registration.torch.library, "Library", lambda *a: library
    )
    monkeypatch.setitem(
        sys.modules,
        "flag_gems",
        SimpleNamespace(
            only_enable=lambda lib, **kw: lib.impl("ones", Mock(), "PrivateUse1")
        ),
    )
    install, emit = Mock(), Mock()
    monkeypatch.setattr(native, "install_collectives", install)
    monkeypatch.setattr(native, "_emit", emit)
    sglang_fl.load_plugin()
    warmup.assert_called_once_with()
    install.assert_not_called()
    assert [call.args[0] for call in emit.call_args_list] == [
        "warmup_compat_ready",
        "plugin_ready",
    ]
    assert emit.call_args.kwargs["flagcx_enabled"] is False


NPU = SimpleNamespace(type="npu", index=0)


class Tensor:
    """NPU-shaped interface backed by real CPU storage for pointer ABI tests."""

    device = NPU
    is_cpu = False

    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        return getattr(self.data, key)

    def copy_(self, other):
        self.data.copy_(other.data)
        return self

    def reshape(self, *args):
        return Tensor(self.data.reshape(*args))

    def movedim(self, *args):
        return Tensor(self.data.movedim(*args))


class CAPI:
    """Fake checked C API; writes actual CPU buffers and records ABI arguments."""

    def __init__(self):
        self.calls = []

    def _buffers(self, api, source, dest, n, *args):
        self.calls.append((api, n, args))
        source_n = n * 2 if api == "rs" else n
        dest_n = n * 2 if api == "ag" else n
        source = torch.frombuffer(
            (ctypes.c_float * source_n).from_address(source.value), dtype=torch.float32
        )
        dest = torch.frombuffer(
            (ctypes.c_float * dest_n).from_address(dest.value), dtype=torch.float32
        )
        if api == "ar":
            dest.copy_(source * 2)
        elif api == "ag":
            dest.copy_(torch.cat((source, source + 10)))
        else:
            dest.copy_(source[:n] * 2)

    def flagcxAllReduce(self, *args):
        self._buffers("ar", *args)

    def flagcxAllGather(self, *args):
        self._buffers("ag", *args)

    def flagcxReduceScatter(self, *args):
        self._buffers("rs", *args)

    def flagcxCommDestroy(self, comm):
        self.calls.append(("destroy", comm))


@pytest.fixture
def routed(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_FL_PROOF_DIR", str(tmp_path))
    monkeypatch.setattr(
        native,
        "torch",
        SimpleNamespace(
            empty=lambda shape, dtype, device: Tensor(torch.empty(shape, dtype=dtype))
        ),
    )
    monkeypatch.setattr(
        flagcx,
        "torch",
        SimpleNamespace(
            empty_like=lambda tensor: Tensor(torch.empty_like(tensor.data)),
            distributed=torch.distributed,
        ),
    )
    api = CAPI()

    def factory(group, device):
        comm = flagcx.FlagCXCommunicator.__new__(flagcx.FlagCXCommunicator)
        comm.disabled, comm.available = factory.disabled, not factory.disabled
        comm.device, comm.flagcx, comm.comm = device, api, 17
        comm._buffer_type = ctypes.c_void_p
        comm._dtype_enum = SimpleNamespace(from_torch=lambda dtype: 7)
        comm._redop_enum = SimpleNamespace(from_torch=lambda op: 0)
        comm._get_stream = lambda: 23
        comm._stream_cache = {}
        return comm

    factory.disabled = False

    class Group:
        def __init__(self, name="tp:0", world_size=2, device=NPU):
            self.unique_name, self.world_size, self.device = name, world_size, device
            self.ranks, self.rank_in_group, self.cpu_group = [8, 9], 0, "gloo"
            self.npu_communicator = SimpleNamespace(
                disabled=False,
                all_reduce=lambda x: ("native_npu", x),
                all_gather=lambda x, dim=-1: ("native_npu", x, dim),
                quant_all_reduce=lambda x: ("native_quant", x),
            )

        def all_reduce(self, input_):
            return ("native", input_)

        def _all_gather_into_tensor(self, output, input):
            return "native_ag"

        def _reduce_scatter_tensor(self, output, input):
            return "native_rs"

        def all_gather(self, input_, dim=-1, output_tensor_list=None):
            return ("native_ag", dim, output_tensor_list)

        def all_gatherv(self, input_, sizes=None, output=None):
            return output

        def destroy(self):
            api.calls.append(("native_destroy",))

    untouched = Group.all_gatherv
    Group.test_factory = factory
    native.install_collectives(Group, factory)
    native.install_collectives(Group, Mock(side_effect=AssertionError("duplicate")))
    assert Group.all_gatherv is untouched
    return Group, api, tmp_path


@pytest.mark.parametrize("direct_npu", [False, True])
def test_real_wrapper_allreduce_abi_identity_and_proof(routed, direct_npu):
    Group, api, directory = routed
    group = Group()
    x = Tensor(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    target = group.npu_communicator if direct_npu else group
    assert target.all_reduce(x) is x
    torch.testing.assert_close(x.data, torch.arange(6).reshape(2, 3).float() * 2)
    assert api.calls == [("ar", 6, (7, 0, 17, 23))]
    records = [
        json.loads(line)
        for path in directory.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [r["event"] for r in records] == [
        "communicator_ready",
        "collective_submitted",
    ]
    assert records[-1]["api"] == "flagcxAllReduce"
    assert records[-1]["ranks"] == [8, 9]


@pytest.mark.parametrize("dim", [-2, -1, 0, 1])
@pytest.mark.parametrize("direct_npu", [False, True])
def test_gather_shape_and_order(routed, dim, direct_npu):
    Group, api, _ = routed
    group = Group()
    data = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    target = group.npu_communicator if direct_npu else group
    actual = target.all_gather(Tensor(data), dim=dim)
    torch.testing.assert_close(actual.data, torch.cat((data, data + 10), dim=dim))
    assert api.calls == [("ag", 6, (7, 17, 23))]


def test_tensor_gather_scatter_and_teardown(routed):
    Group, api, _ = routed
    group = Group()
    data = Tensor(torch.arange(6, dtype=torch.float32))
    gathered = Tensor(torch.empty(12))
    assert group._all_gather_into_tensor(gathered, input=data) is None
    torch.testing.assert_close(gathered.data, torch.cat((data.data, data.data + 10)))
    reduced = Tensor(torch.empty(3))
    assert group._reduce_scatter_tensor(reduced, input=data) is reduced
    torch.testing.assert_close(reduced.data, data.data[:3] * 2)
    group.destroy()
    assert api.calls[-2:] == [("destroy", 17), ("native_destroy",)]


@pytest.mark.parametrize(
    "name,size", [("world:0", 2), ("moe_ep:0", 2), ("pp:0", 2), ("tp:0", 1)]
)
def test_other_groups_and_tp1_untouched(routed, name, size):
    Group, api, _ = routed
    group = Group(name=name, world_size=size)
    assert group.all_reduce(Tensor(torch.zeros(2)))[0] == "native"
    assert group._fl_native_comm is None
    assert api.calls == []


def test_cpu_lists_quant_and_variable_outputs_stay_native(routed):
    Group, api, _ = routed
    group = Group()
    cpu = torch.zeros(2)
    assert group.all_reduce(cpu)[0] == "native"
    assert group.npu_communicator.all_reduce(cpu)[0] == "native_npu"
    assert group.npu_communicator.all_gather(cpu)[0] == "native_npu"
    assert group.npu_communicator.quant_all_reduce(cpu)[0] == "native_quant"
    outputs = [object(), object()]
    assert group.all_gather(Tensor(cpu), output_tensor_list=outputs)[2] is outputs
    assert group.all_gatherv(cpu, output=outputs) is outputs
    assert api.calls == []


def test_no_fallback_when_scoped_comm_disabled_or_missing(routed):
    Group, api, _ = routed
    group = Group()
    group._fl_native_comm.disabled = True
    with pytest.raises(RuntimeError, match="unavailable"):
        group.all_reduce(Tensor(torch.zeros(2)))
    assert api.calls == []


def test_failed_initialization_aborts_before_native_fallback(routed):
    Group, api, _ = routed
    Group.test_factory.disabled = True
    with pytest.raises(RuntimeError, match="initialization failed"):
        Group()
    assert api.calls == [("destroy", 17)]


def test_wrapper_must_resolve_inside_selected_source_tree(environment, monkeypatch):
    wrapper = SimpleNamespace(
        FLAGCXLibrary=object,
        buffer_type=object,
        flagcxComm_t=object,
        flagcxDataTypeEnum=object,
        flagcxRedOpTypeEnum=object,
        flagcxUniqueId=object,
        __file__="/other/FlagCX/plugin/interservice/flagcx_wrapper.py",
    )
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", wrapper)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(RuntimeError, match="path mismatch"):
        flagcx._import_flagcx_wrapper()
    wrapper.__file__ = str(environment / "plugin/interservice/flagcx_wrapper.py")
    assert flagcx._import_flagcx_wrapper()[0] is object


def test_invalid_buffer_and_dimension_rejected_before_capi(routed):
    Group, api, _ = routed
    group = Group()
    with pytest.raises(ValueError, match="contiguous"):
        group.all_reduce(Tensor(torch.empty(2, 3).t()))
    with pytest.raises(ValueError, match="output"):
        group._all_gather_into_tensor(Tensor(torch.empty(2)), Tensor(torch.empty(2)))
    with pytest.raises(ValueError, match="equal splits"):
        group._reduce_scatter_tensor(Tensor(torch.empty(2)), Tensor(torch.empty(3)))
    with pytest.raises(ValueError, match="dimension"):
        group.all_gather(Tensor(torch.empty(2)), dim=1)
    assert api.calls == []


def test_failed_capi_call_does_not_emit_proof(routed):
    Group, api, directory = routed
    group = Group()
    api.flagcxAllReduce = Mock(side_effect=RuntimeError("transport failed"))
    with pytest.raises(RuntimeError, match="transport failed"):
        group.all_reduce(Tensor(torch.empty(2)))
    assert "collective_submitted" not in next(directory.glob("*.jsonl")).read_text()


def test_bad_signature_does_not_partially_patch(routed):
    Group, _, _ = routed

    class Changed(Group):
        _fl_native_installed = False

        def all_gather(self, input_, changed=False):
            pass

    original = Changed.__init__
    with pytest.raises(RuntimeError, match="signature"):
        native.install_collectives(Changed)
    assert Changed.__init__ is original


def test_reference_sg_signatures_and_npu_routing(monkeypatch, routed):
    """Execute selected real SG method bodies from source, without importing SG."""
    source = os.environ.get("SGLANG_FL_SG_SOURCE")
    if not source:
        pytest.skip("Set SGLANG_FL_SG_SOURCE to an audited SG python/sglang directory")
    path = Path(source) / "srt/distributed/parallel_state.py"
    tree = ast.parse(path.read_text())
    coordinator = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GroupCoordinator"
    )
    Group, api, _ = routed
    for name in (
        "all_reduce",
        "all_gather",
        "_all_gather_into_tensor",
        "_reduce_scatter_tensor",
        "destroy",
    ):
        method = next(
            n
            for n in coordinator.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        expected = list(inspect.signature(getattr(Group, name)).parameters)
        assert [a.arg for a in method.args.args] == expected
    method = next(
        n
        for n in coordinator.body
        if isinstance(n, ast.FunctionDef) and n.name == "all_reduce"
    )
    module = ast.Module(body=[method], type_ignores=[])
    scope = {"torch": torch}
    exec(compile(module, str(path), "exec"), scope)
    group = Group()
    group.hpu_communicator = group.xpu_communicator = None
    x = Tensor(torch.tensor([3.0]))
    # Original SG GroupCoordinator -> original NpuCommunicator instance entry
    # (scoped binding) -> real plugin FlagCX wrapper -> fake C ABI on CPU.
    assert scope["all_reduce"](group, x) is x
    assert x.data.item() == 6.0
    assert api.calls[0][0] == "ar"
    method = next(
        n
        for n in coordinator.body
        if isinstance(n, ast.FunctionDef) and n.name == "_reduce_scatter_tensor"
    )
    scope = {
        "torch": SimpleNamespace(
            Tensor=torch.Tensor,
            distributed=SimpleNamespace(reduce_scatter_tensor=Mock()),
        )
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    output = Tensor(torch.empty(1))
    group.pynccl_comm, group.device_group = None, "native_device_group"
    assert scope["_reduce_scatter_tensor"](group, output, x) is output
