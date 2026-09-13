"""Share GLM's immutable physical RoPE placeholder across attention layers.

Each per-layer view remains contiguous for the CANN attention ABI. The existing
zero-RoPE setter writes only latent K; disaggregation and prefix caching are
excluded from this explicit option.
"""
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import torch
from torch.overrides import TorchFunctionMode

_SCOPE = ContextVar('glm53_shared_zero_rope', default=False)
_PATCHED = False


class SharedZeroMode(TorchFunctionMode):
    def __init__(self, pages, page_size=64):
        super().__init__()
        self.pages, self.page_size = pages, page_size
        self.redirected = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if getattr(func, '__name__', '') == 'zeros' and args:
            shape = args[0]
            if (isinstance(shape, (tuple, list, torch.Size)) and len(shape) == 5
                    and shape[0] > 1 and tuple(shape[1:]) == (self.pages, self.page_size, 1, 64)):
                if self.redirected:
                    raise RuntimeError('Unexpected second GLM zero-RoPE allocation')
                self.redirected += 1
                value = func((1, *shape[1:]), *args[1:], **kwargs)
                return value.expand(*shape)
        return func(*args, **kwargs)


def supported(runner):
    args = runner.server_args
    arch = getattr(runner.model_config.hf_config, 'architectures', ()) or ()
    if not any(a in ('Glm5NextForConditionalGeneration',
                     'Glm5NextForConditionalGenerationNextN') for a in arch):
        return False
    if not (str(args.device).startswith('npu') and args.quantization == 'modelslim'
            and args.disable_radix_cache and args.disaggregation_mode == 'null'
            and args.tp_size == args.ep_size and args.tp_size in (16, 32)
            and args.nnodes == args.tp_size // 16 and args.pp_size == 1
            and not args.enable_dp_attention and not args.enable_two_batch_overlap
            and args.page_size == 64):
        raise ValueError('Shared GLM zero RoPE requires TP16/32 without prefix cache/disaggregation')
    return True


def patch_shared_zero_rope():
    global _PATCHED
    if _PATCHED:
        return
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    original_pools = ModelRunnerKVCacheMixin._init_pools
    original_init = NPUMLATokenToKVPool.__init__
    original_bytes = NPUMLATokenToKVPool.get_kv_size_bytes

    @wraps(original_pools)
    def init_pools(self, *args, **kwargs):
        enabled = os.getenv('SGLANG_FL_GLM53_SHARE_ZERO_ROPE', '0') == '1'
        token = _SCOPE.set(enabled and supported(self))
        try:
            return original_pools(self, *args, **kwargs)
        finally:
            _SCOPE.reset(token)

    @wraps(original_init)
    def init(self, *args, **kwargs):
        if not _SCOPE.get():
            return original_init(self, *args, **kwargs)
        def arg(name, index):
            return kwargs[name] if name in kwargs else args[index]
        if arg('qk_rope_head_dim', 4) != 0 or arg('kv_lora_rank', 3) != 512:
            raise ValueError('Shared RoPE storage requires logical zero RoPE and latent width 512')
        pages = arg('size', 0) // arg('page_size', 1) + 1
        layers = arg('layer_num', 6)
        self._glm53_shared_zero_rope = layers > 1
        mode = SharedZeroMode(pages, arg('page_size', 1))
        with mode:
            original_init(self, *args, **kwargs)
        v = self.v_buffer
        if layers > 1 and (mode.redirected != 1 or v.stride(0) != 0
                           or not all(v[i].is_contiguous() for i in range(layers))):
            raise RuntimeError('Invalid shared zero-RoPE per-layer storage')
        if not self._sglang_fl_zero_rope:
            raise RuntimeError('Shared placeholder requires the zero-RoPE-only cache setter')
        if base := os.getenv('SGLANG_FL_GLM53_AUDIT_DIR'):
            folder = Path(base).parent / 'zero-rope'
            folder.mkdir(parents=True, exist_ok=True)
            row = dict(layers=layers, pages=pages, dtype=str(v.dtype),
                logical_bytes=v.numel()*v.element_size(),
                storage_bytes=v.untyped_storage().nbytes(),
                saved_bytes=v.numel()*v.element_size()-v.untyped_storage().nbytes(),
                shared=self._glm53_shared_zero_rope, per_layer_contiguous=True)
            with (folder / f'{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')

    @wraps(original_bytes)
    def size_bytes(self):
        total = original_bytes(self)
        if getattr(self, '_glm53_shared_zero_rope', False):
            v = self.v_buffer
            total -= v.numel()*v.element_size()-v.untyped_storage().nbytes()
        return total

    ModelRunnerKVCacheMixin._init_pools = init_pools
    NPUMLATokenToKVPool.__init__ = init
    NPUMLATokenToKVPool.get_kv_size_bytes = size_bytes
    _PATCHED = True
