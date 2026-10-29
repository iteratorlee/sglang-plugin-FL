"""Audited source contracts; real Triton JIT checks are CPU-only when available."""

import ast
import hashlib
import importlib.util
import inspect
import linecache
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).with_name("fixtures") / "native_sampling_target_only.py"
spec = importlib.util.spec_from_file_location(
    "native_sampling_cpu", ROOT / "sglang_fl/native_sampling.py"
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)  # Avoid the plugin __init__ and its torch import.


class CPUJIT:
    """Only the inspected ABI; this double cannot execute or compile a kernel."""

    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        self.arg_names = list(self.signature.parameters)
        self.constexprs = [
            i
            for i, p in enumerate(self.signature.parameters.values())
            if p.annotation is fn.__globals__["tl"].constexpr
        ]
        source = inspect.getsource(fn)
        self.src = ast.get_source_segment(source, ast.parse(source).body[0]) + "\n"


@pytest.fixture(params=["double", "triton"])
def audited(request, monkeypatch):
    if request.param == "triton":
        triton = pytest.importorskip("triton")
        import triton.language as tl
    else:
        tl = ModuleType("triton.language")
        tl.constexpr = object()
        triton = ModuleType("triton")
        triton.language, triton.JITFunction, triton.jit = tl, CPUJIT, CPUJIT
        monkeypatch.setitem(sys.modules, "triton", triton)
        monkeypatch.setitem(sys.modules, "triton.language", tl)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Tensor=type("Tensor", (), {}))
    )
    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel_npu.utils.triton_utils",
        SimpleNamespace(get_device_properties=Mock()),
    )
    spec = importlib.util.spec_from_file_location(patch.TARGET, SNAPSHOT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, patch.TARGET, module)
    spec.loader.exec_module(module)
    return module, triton


def test_exact_snapshot_one_operator_abi_aliases_and_idempotence(audited):
    module, triton = audited
    raw = SNAPSHOT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == patch.SOURCE_SHA256
    before = dict(vars(module))
    old = getattr(module, patch.KERNEL)
    old_key = old.cache_key if triton.JITFunction is not CPUJIT else None
    alias = getattr(module, patch.ENTRY)
    assert patch.configure()["status"] == "patched"
    new = getattr(module, patch.KERNEL)
    assert old is not new
    assert getattr(module, patch.ENTRY) is alias
    assert alias.__globals__[patch.KERNEL] is new
    assert [k for k in vars(module) if vars(module)[k] is not before[k]] == [
        patch.KERNEL
    ]
    assert inspect.signature(new.fn) == inspect.signature(old.fn)
    assert new.arg_names == old.arg_names
    assert new.constexprs == old.constexprs == [13, 14, 15]
    assert new.fn.__globals__ is vars(module)
    assert new.src.strip() == old.src.strip().replace("coin <=", "coin <", 1)
    expected = ast.parse(old.src)
    comparisons = [
        n
        for n in ast.walk(expected)
        if isinstance(n, ast.Compare) and any(isinstance(op, ast.LtE) for op in n.ops)
    ]
    assert len(comparisons) == 1
    comparisons[0].ops = [ast.Lt()]
    assert ast.dump(expected) == ast.dump(ast.parse(new.src))
    linecache.checkcache()
    assert inspect.getsource(new.fn).strip() == new.src.strip()
    assert new.fn.__code__.co_filename.startswith("<sglang_fl.native_sampling.")
    assert patch.install(module) == "already_installed"
    assert getattr(module, patch.KERNEL) is new
    assert SNAPSHOT.read_bytes() == raw
    if triton.JITFunction is not CPUJIT:
        # Real public JIT construction/hash traversal, without a device/driver.
        assert old.cache_key == old_key
        assert new.cache_key != old_key
        assert triton.jit(new.fn).cache_key == new.cache_key


def acceptance(source):
    value = next(
        n.value
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "accepted" for t in n.targets)
    )
    code = compile(ast.Expression(value), "<acceptance-oracle>", "eval")
    return lambda **values: eval(code, {"__builtins__": {}}, values)


def test_zero_boundary_and_threshold_semantics(audited):
    module, _ = audited
    old = acceptance(getattr(module, patch.KERNEL).src)
    patch.install(module)
    new = acceptance(getattr(module, patch.KERNEL).src)
    values = dict(
        coin=0.0,
        prob_acc=0.0,
        target_prob_single=0.0,
        threshold_acc=1.0,
        threshold_single=1.0,
    )
    assert old(**values) and not new(**values)
    # The explicit single-token threshold shortcut retains its original meaning.
    assert new(**dict(values, threshold_single=0.0))
    for single in (0.0, 0.2, 1.0):
        for acc in (1e-9, 0.25, 1.0):
            for prob in (0.0, 0.125, 0.5, 1.0):
                for coin in (0.0, 0.03125, 0.125, 0.5, 0.99):
                    values.update(
                        coin=coin,
                        prob_acc=prob,
                        target_prob_single=prob,
                        threshold_single=single,
                        threshold_acc=acc,
                    )
                    if coin != prob / acc or prob >= single:
                        assert old(**values) == new(**values)
                    else:
                        assert old(**values) and not new(**values)


@pytest.mark.parametrize(
    "change",
    ["file", "src", "args", "constexpr", "signature", "defaults", "entry", "marker"],
)
def test_unknown_versions_and_abi_fail_closed(audited, tmp_path, change):
    module, triton = audited
    old = getattr(module, patch.KERNEL)
    if change == "file":
        unknown = tmp_path / "unknown.py"
        unknown.write_bytes(SNAPSHOT.read_bytes() + b"\n# unknown revision\n")
        module.__file__ = str(unknown)
    elif change == "src":
        changed = old.src.replace("num_accepted = 0", "num_accepted = 1")
        unknown = tmp_path / "unknown_kernel.py"
        unknown.write_text(changed)
        scope = {}
        exec(compile(changed, str(unknown), "exec"), vars(module), scope)
        old = triton.jit(scope[patch.KERNEL])
        setattr(module, patch.KERNEL, old)
    elif change == "args":
        old.arg_names = old.arg_names[:-1]
    elif change == "constexpr":
        old.constexprs = [14, 15]
    elif change == "signature":
        old.signature = inspect.Signature()
    elif change == "defaults":
        old.fn.__defaults__ = (1,)
    elif change == "entry":
        getattr(module, patch.ENTRY).__defaults__ = (0.0, 1.0, True)
    elif change == "marker":
        old._fl_native_sampling = patch.SOURCE_SHA256
    before = dict(vars(module))
    with pytest.raises(RuntimeError, match="Unsupported"):
        patch.install(module)
    assert vars(module) == before
    assert getattr(module, patch.KERNEL) is old


def test_jit_construction_failure_is_atomic(audited, monkeypatch):
    module, triton = audited
    before = dict(vars(module))
    monkeypatch.setattr(triton, "jit", Mock(side_effect=RuntimeError("JIT failure")))
    with pytest.raises(RuntimeError, match="JIT failure"):
        patch.install(module)
    assert vars(module) == before


def test_repeat_checks_modified_jit_instead_of_trusting_marker(audited):
    module, _ = audited
    patch.install(module)
    getattr(module, patch.KERNEL).arg_names = []
    with pytest.raises(RuntimeError, match="signature/ABI"):
        patch.install(module)


def test_concurrent_initialization_builds_one_jit(audited, monkeypatch):
    module, triton = audited
    factory = Mock(wraps=triton.jit)
    monkeypatch.setattr(triton, "jit", factory)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: patch.install(module), range(8)))
    assert results.count("patched") == 1
    assert results.count("already_installed") == 7
    factory.assert_called_once()


@pytest.mark.parametrize(
    "bad",
    [
        "coin < prob_acc / threshold_acc",
        "coin <= threshold_acc",
        "coin <= prob_acc / threshold_acc\nx = a <= b",
    ],
)
def test_ast_guard_refuses_other_or_multiple_comparisons(bad):
    with pytest.raises(RuntimeError, match="comparison"):
        patch._strict_source(bad)


@pytest.mark.parametrize("fail", [False, True])
def test_native_installs_before_warmup_and_propagates_failure(monkeypatch, fail):
    source = ast.parse((ROOT / "sglang_fl/native.py").read_text())
    load = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_native_plugin"
    )
    events = []

    def configure():
        events.append("sampling")
        if fail:
            raise RuntimeError("unsupported sampling")
        return {"status": "patched"}

    def allocator():
        events.append("allocator")
        raise RuntimeError("stop before warmup")

    monkeypatch.setitem(
        sys.modules, "sglang_fl.native_sampling", SimpleNamespace(configure=configure)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_fl.native_allocator_warmup",
        SimpleNamespace(configure=allocator),
    )
    scope = dict(
        _flagcx_enabled=lambda: False,
        validate_environment=lambda: ({}, []),
        _emit=Mock(),
    )
    exec(
        compile(
            ast.Module(body=[load], type_ignores=[]), "<native-integration>", "exec"
        ),
        scope,
    )
    with pytest.raises(
        RuntimeError, match="unsupported sampling" if fail else "stop before warmup"
    ):
        scope["load_native_plugin"]()
    assert events == (["sampling"] if fail else ["sampling", "allocator"])
