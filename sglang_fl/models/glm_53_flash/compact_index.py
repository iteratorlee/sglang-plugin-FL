"""Optional request-owned storage for the four-token KPool index cache.

Attention K/V retain their allocator and physical page table. Only compressed
index keys use this layout. Prefix caching and disaggregation are unsupported.
"""
from dataclasses import dataclass, asdict
from functools import wraps
import json
import os
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode


@dataclass(frozen=True)
class IndexLayout:
    requests: int
    pages_per_request: int
    full_pages: int

    @property
    def pages(self):
        return 1 + self.requests * self.pages_per_request


def make_layout(requests, context, full_tokens, page_size=64, draft_tokens=0):
    if page_size != 64 or min(requests, context, full_tokens) <= 0:
        raise ValueError('Compact GLM index requires positive capacities and page size 64')
    # Reserve the upstream speculative extension plus one compressed guard page.
    per_request = (context + 4 + draft_tokens + 255) // 256 + 1
    layout = IndexLayout(requests, per_request, full_tokens // 64 + 1)
    return layout if layout.pages < layout.full_pages else None


class IndexAllocationMode(TorchFunctionMode):
    """Intercept one scoped index allocation, without allocating the large tensor."""
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.redirected = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if getattr(func, '__name__', '') == 'zeros' and args:
            shape = args[0]
            if (isinstance(shape, (tuple, list, torch.Size)) and len(shape) == 5
                    and shape[0] > 0 and shape[1] == self.layout.full_pages
                    and tuple(shape[2:]) == (64, 1, 128)):
                if self.redirected:
                    raise RuntimeError('Unexpected second GLM index allocation')
                args = ((shape[0], self.layout.pages, *shape[2:]), *args[1:])
                self.redirected += 1
        return func(*args, **kwargs)


def index_block_table(batch, original):
    pool = batch.token_to_kv_pool
    full_pool = getattr(pool, 'full_kv_pool', pool)
    layout = getattr(full_pool, '_glm53_index_layout', None)
    if layout is None:
        return original
    from .compact_index_npu import request_block_table
    return request_block_table(batch.req_pool_indices, original, layout)


def patch_compact_index():
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
    original = ModelRunnerKVCacheMixin._init_pools
    if getattr(original, '_glm53_compact_index', False):
        return

    @wraps(original)
    def init_pools(self, *args, **kwargs):
        if os.getenv('SGLANG_FL_GLM53_COMPACT_INDEX', '0') != '1':
            return original(self, *args, **kwargs)
        cfg = self.server_args
        arch = getattr(self.model_config.hf_config, 'architectures', ()) or ()
        if not any(a in ('Glm5NextForConditionalGeneration',
                         'Glm5NextForConditionalGenerationNextN') for a in arch):
            return original(self, *args, **kwargs)
        if not (str(cfg.device).startswith('npu') and cfg.quantization == 'modelslim'
                and cfg.disable_radix_cache and cfg.disaggregation_mode == 'null'
                and cfg.tp_size == cfg.ep_size and cfg.tp_size in (16, 32)
                and cfg.nnodes == cfg.tp_size // 16 and cfg.pp_size == 1
                and not cfg.enable_dp_attention and not cfg.enable_two_batch_overlap
                and int(self.model_config.hf_text_config.index_head_dim) == 128):
            raise ValueError('Compact GLM index requires TP16/32, no prefix cache or disaggregation')
        layout = make_layout(self.max_running_requests,
            self.model_config.context_len, self.max_total_num_tokens, cfg.page_size,
            cfg.speculative_num_draft_tokens or 0)
        if layout is None:
            return original(self, *args, **kwargs)
        mode = IndexAllocationMode(layout)
        with mode:
            result = original(self, *args, **kwargs)
        pool = self.token_to_kv_pool.full_kv_pool
        if (mode.redirected != 1 or pool.index_k_buffer.shape[1] != layout.pages
                or self.req_to_token_pool.size != layout.requests):
            raise RuntimeError('Compact GLM index allocation or request capacity mismatch')
        pool._glm53_index_layout = layout
        if base := os.getenv('SGLANG_FL_GLM53_AUDIT_DIR'):
            folder = Path(base).parent / 'compact-index'
            folder.mkdir(parents=True, exist_ok=True)
            row = dict(asdict(layout), pages=layout.pages,
                layers=pool.index_k_buffer.shape[0], dtype=str(pool.index_k_buffer.dtype),
                saved_bytes=(layout.full_pages-layout.pages)*64*128
                    *pool.index_k_buffer.element_size()*pool.index_k_buffer.shape[0])
            with (folder / f'{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        return result

    init_pools._glm53_compact_index = True
    ModelRunnerKVCacheMixin._init_pools = init_pools
