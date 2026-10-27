"""CPU contracts for the plugin's independently audited Library proxy."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_fl import native_registration as registration


@pytest.fixture(autouse=True)
def isolated_retention(monkeypatch):
    monkeypatch.setattr(registration, "_libraries", [])


def test_exact_impl_success_and_logging_options_are_forwarded(monkeypatch):
    library, fn = Mock(), Mock()
    monkeypatch.setattr(
        registration.torch.library, "Library", Mock(return_value=library)
    )
    gems = SimpleNamespace(
        only_enable=Mock(
            side_effect=lambda lib, **kw: lib.impl(
                "ones", fn, "PrivateUse1", with_keyset=False
            )
        )
    )
    audited = registration.enable_ones(gems, record=True, once=False, path="gems.log")
    gems.only_enable.assert_called_once_with(
        lib=audited, include=["ones"], record=True, once=False, path="gems.log"
    )
    library.impl.assert_called_once_with("ones", fn, "PrivateUse1", with_keyset=False)
    assert audited.registered == [("ones", "PrivateUse1")]
    assert audited._library is library
    assert registration._libraries == [audited]


@pytest.mark.parametrize(
    "attempts",
    [
        [],
        [("ones", "CPU")],
        [("ones_like", "PrivateUse1")],
        [("ones", "PrivateUse1"), ("ones", "PrivateUse1")],
        [("ones", "PrivateUse1"), ("zeros", "PrivateUse1")],
        [("zeros", "PrivateUse1"), ("ones", "PrivateUse1")],
    ],
)
def test_silent_noop_and_swallowed_extra_attempts_fail(monkeypatch, attempts):
    library = Mock()
    monkeypatch.setattr(registration.torch.library, "Library", lambda *a: library)

    def only_enable(lib, **kwargs):
        for op, key in attempts:
            try:
                lib.impl(op, Mock(), key)
            except RuntimeError:
                pass

    with pytest.raises(RuntimeError, match="registration incomplete"):
        registration.enable_ones(SimpleNamespace(only_enable=only_enable))
    assert library.impl.call_count == int(
        bool(attempts) and attempts[0] == ("ones", "PrivateUse1")
    )


def test_success_is_recorded_only_after_delegate_returns():
    library = Mock()
    audited = registration.AuditedLibrary(library)

    def rejected(*args, **kwargs):
        assert audited.attempted == [("ones", "PrivateUse1")]
        assert audited.registered == []
        raise RuntimeError("duplicate dispatcher implementation")

    library.impl.side_effect = rejected
    with pytest.raises(RuntimeError, match="duplicate dispatcher"):
        audited.impl("ones", Mock(), "PrivateUse1")
    assert audited.registered == []
    with pytest.raises(RuntimeError, match="registration incomplete"):
        audited.assert_registered()


def test_real_torch_dispatcher_registration_survives_garbage_collection():
    # A fresh CPU-only subprocess registers a real PrivateUse1 kernel but never
    # allocates a PrivateUse1 tensor or imports an accelerator module.
    code = """
import gc
import weakref
from types import SimpleNamespace
import torch
from sglang_fl.native_registration import enable_ones

def ones(*args, **kwargs):
    raise AssertionError('This CPU contract must never execute an NPU kernel')

def only_enable(lib, **kwargs):
    lib.impl('ones', ones, 'PrivateUse1')

audited = enable_ones(SimpleNamespace(only_enable=only_enable), record=False, once=True)
ref = weakref.ref(audited._library)
del audited
gc.collect()
assert ref() is not None
assert torch._C._dispatch_has_kernel_for_dispatch_key('aten::ones', 'PrivateUse1')
print('real dispatcher registration retained')
"""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "real dispatcher registration retained" in result.stdout
