"""Optional, object-local adapter for the old add_rmsnorm_bias launch ABI.

Enable only after observing the missing-constexpr error with FlagTree 3.5.
The native JIT, kernel code, tensors, cache key, grid and outputs are retained.
No source files are written and no global Triton class is patched.
"""

import functools
import importlib


class _CompiledKernel:
    def __init__(self, kernel, constants):
        self._kernel = kernel
        self._constants = constants

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def __getitem__(self, grid):
        launch = self._kernel[grid]

        def run(*args, **kwargs):
            # Old SGKernel launches omit the five constexpr values it passed
            # to warmup; newer fixed callers already supply them.
            if len(args) == 9 and not kwargs:
                return launch(*args, *self._constants)
            return launch(*args, **kwargs)

        return run


def install(module=None):
    if module is None:
        module = importlib.import_module("sgl_kernel_npu.norm.add_rmsnorm_bias")
    jit = module.add_rmsnorm_bias_kernel
    if getattr(jit, "_fl_native_warmup", False):
        return
    names = tuple(jit.arg_names)
    if names != (
        "input_ptr",
        "residual_ptr",
        "norm_weight_ptr",
        "norm_bias_ptr",
        "quant_scale_ptr",
        "quant_offset_ptr",
        "output_ptr",
        "output2_ptr",
        "batch_size",
        "hidden_size",
        "eps",
        "BLOCK_SIZE",
        "COL_BLOCK_SIZE",
        "SCALE",
    ):
        raise RuntimeError(f"Unsupported add_rmsnorm_bias kernel ABI: {names}")
    if module.kernels:
        raise RuntimeError(
            "Install warmup compatibility before add_rmsnorm_bias cache population"
        )
    warmup = jit.warmup

    @functools.wraps(warmup)
    def adapted_warmup(*args, **kwargs):
        if len(args) != 14:
            raise RuntimeError(
                "Expected the audited 14-argument add_rmsnorm_bias warmup"
            )
        return _CompiledKernel(warmup(*args, **kwargs), args[9:14])

    jit.warmup = adapted_warmup
    jit._fl_native_warmup = True
