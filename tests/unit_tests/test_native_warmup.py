"""CPU ABI contract for the optional native SGKernel warmup adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_fl.native_warmup import install


def module():
    args = (
        "input_ptr residual_ptr norm_weight_ptr norm_bias_ptr quant_scale_ptr "
        "quant_offset_ptr output_ptr output2_ptr batch_size hidden_size eps "
        "BLOCK_SIZE COL_BLOCK_SIZE SCALE"
    ).split()
    launch = Mock(return_value=("native-output", "native-residual"))

    class Kernel:
        _init_handles = Mock()

        def __getitem__(self, grid):
            assert grid == (3, 1, 1)
            return launch

    kernel = Kernel()
    jit = SimpleNamespace(arg_names=args, warmup=Mock(return_value=kernel))
    return SimpleNamespace(add_rmsnorm_bias_kernel=jit, kernels={}), kernel, launch


def test_only_missing_launch_constants_are_forwarded():
    mod, kernel, launch = module()
    jit = mod.add_rmsnorm_bias_kernel
    original = jit.warmup
    install(mod)
    installed = jit.warmup
    install(mod)
    assert jit.warmup is installed
    args = tuple(object() for _ in range(9)) + (4096, 1e-6, 4096, 2048, False)
    compiled = jit.warmup(*args, grid=(3,))
    original.assert_called_once_with(*args, grid=(3,))
    compiled._init_handles()
    kernel._init_handles.assert_called_once_with()
    assert compiled[(3, 1, 1)](*args[:9]) == ("native-output", "native-residual")
    launch.assert_called_once_with(*args)
    launch.reset_mock()
    compiled[(3, 1, 1)](*args)
    launch.assert_called_once_with(*args)


@pytest.mark.parametrize("bad", ["signature", "cache"])
def test_unknown_abi_or_populated_cache_rejected_without_mutation(bad):
    mod, _, _ = module()
    original = mod.add_rmsnorm_bias_kernel.warmup
    if bad == "signature":
        mod.add_rmsnorm_bias_kernel.arg_names = ["different"]
    else:
        mod.kernels["existing"] = object()
    with pytest.raises(RuntimeError):
        install(mod)
    assert mod.add_rmsnorm_bias_kernel.warmup is original
