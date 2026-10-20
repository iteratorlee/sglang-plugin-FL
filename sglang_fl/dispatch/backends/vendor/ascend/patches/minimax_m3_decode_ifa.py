# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Opt-in fixed-metadata sparse decode IFA for the measured bs16 shape.

The virtual sequence always contains 17 pages. Device masks retain slot order,
duplicates, partial pages and empty rows. Host lengths never depend on requests.
Only plain decode/idle uses this path; smaller batches retain native Triton.
"""
import __future__
import ast
import functools
import hashlib
import inspect
import logging
import os
from pathlib import Path
import textwrap
from types import CodeType, FunctionType

import torch
import torch_npu
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_SOURCES = {
    'backend': '2798456cbe4f907fbbccf1eaab1114d2324e79bf1205adb6d4c2e9b8b738b048',
    'attention': 'c60257620ddf5dc8f2fc3a5f07a4509ecde308200fafe35c9c02ed166cb4fc9d',
    'graphs': '5ed98b6b5bbbaa162513801e877a24671b58ec2b8eaba3c8d446357efe0e5dc1',
    'dispatch': 'f292b98c7786cfe978610c78814a48d71d83aebfd97b859b9d2e91d946d07168',
}
_HELPER = '_sglang_fl_minimax_fixed_decode_ifa'
_ORIGINAL = None
_backend_module = None
_IFA_OP = None
_IFA_DEFAULT = None
_logged = set()


@triton.jit
def _prepare(
    topk, seq, rows, rtt, bt, mask, any_token,
    topk_b, topk_s, seq_s, rows_s, rtt_r, rtt_c,
    ROWS: tl.constexpr, COLS: tl.constexpr, PAGES: tl.constexpr,
):
    b = tl.program_id(0)
    slots = tl.arange(0, 32)
    tokens = tl.arange(0, 128)
    logical = tl.load(topk + b * topk_b + slots * topk_s, slots < 17, other=-1)
    length = tl.minimum(tl.maximum(tl.load(seq + b * seq_s), 0), COLS).to(tl.int32)
    row = tl.load(rows + b * rows_s).to(tl.int64)
    valid_logical = (slots < 17) & (logical >= 0) & (logical < COLS // 128)
    safe_logical = tl.where(valid_logical, logical, 0)
    valid_row = (row >= 0) & (row < ROWS)
    safe_row = tl.where(valid_row, row, 0)
    valid = valid_logical & valid_row & (safe_logical * 128 < length)
    # Only the 17 address lanes need INT64. Widening the 17x128 token mask
    # creates expensive emulated wide arithmetic on Ascend vector cores.
    physical_slot = tl.load(rtt + safe_row * rtt_r + safe_logical.to(tl.int64) * 128 * rtt_c,
                            valid, other=-128)
    page = physical_slot // 128
    valid = valid & (physical_slot >= 0) & (page < PAGES)
    tl.store(bt + b * 17 + slots, tl.where(valid, page, 0), slots < 17)
    attend = valid[:, None] & (safe_logical[:, None] * 128 + tokens[None, :] < length)
    tl.store(mask + b * 2176 + slots[:, None] * 128 + tokens[None, :],
             ~attend, slots[:, None] < 17)
    tl.store(any_token + b, tl.sum(valid.to(tl.int32), 0) > 0)


def fixed_sparse_decode(q, k, v, topk, seq, req_to_token, req_pool_indices, scale=None):
    """Device-gate helper; rejects unknown mode stacks before device work."""
    from torch._ops import _len_torch_dispatch_stack_pre_dispatch
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    from torch_npu.npu.graphs import _GraphDispatchMode

    stack = _get_current_dispatch_mode_stack()
    if _len_torch_dispatch_stack_pre_dispatch() or (
        stack and (len(stack) != 1 or type(stack[0]) is not _GraphDispatchMode)
    ):
        raise RuntimeError('Unsupported dispatch stack for fixed IFA')
    tensors = (q, k, v, topk, seq, req_to_token, req_pool_indices)
    assert all(type(t) is torch.Tensor and t.device == q.device for t in tensors)
    b = q.shape[0]
    assert q.shape == (b, 16, 128) and 1 <= b <= 16 and q.dtype == torch.bfloat16
    assert q.is_contiguous() and k.shape[0] >= 17 * b
    assert k.numel() <= 2**31 - 1
    assert k.shape == v.shape and k.shape[1:] == (128, 1, 128)
    assert k.dtype == v.dtype == q.dtype and k.is_contiguous() and v.is_contiguous()
    assert topk.shape == (1, b, 17) and topk.dtype == torch.int32
    assert seq.shape == req_pool_indices.shape == (b,)
    assert req_to_token.ndim == 2 and 128 <= req_to_token.shape[1] <= 2**31 - 1
    assert all(t.dtype in (torch.int32, torch.int64) for t in (seq, req_to_token, req_pool_indices))
    bt = torch.empty((b, 17), dtype=torch.int32, device=q.device)
    mask = torch.empty((b, 1, 1, 2176), dtype=torch.bool, device=q.device)
    any_token = torch.empty((b,), dtype=torch.bool, device=q.device)
    _prepare[(b,)](
        topk, seq, req_pool_indices, req_to_token, bt, mask, any_token,
        topk.stride(1), topk.stride(2), seq.stride(0), req_pool_indices.stride(0),
        req_to_token.stride(0), req_to_token.stride(1),
        ROWS=req_to_token.shape[0], COLS=req_to_token.shape[1], PAGES=k.shape[0],
        num_warps=4, num_stages=1,
    )
    args = (q.view(b, 16, 1, 128), k.view(k.shape[0], 128, 128), v.view(v.shape[0], 128, 128))
    op = _IFA_OP if _IFA_OP is not None else torch.ops.npu.npu_incre_flash_attention
    output = op(
        *args, atten_mask=mask, block_table=bt, block_size=128,
        num_heads=16, num_key_value_heads=1, input_layout='BNSD',
        scale_value=128**-0.5 if scale is None else scale,
        actual_seq_lengths=[2176] * b, inner_precise=0,
    )
    return torch.where(any_token[:, None, None], output.view(b, 16, 128), 0)


def _dispatch(*, _original, _backend, _forward_batch, _idx_q, _idx_v_cache, **kw):
    from torch._ops import _len_torch_dispatch_stack_pre_dispatch
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
    from torch_npu.npu.graphs import _GraphDispatchMode

    q, k, v = kw['q'], kw['k_cache_bnsd'], kw['v_cache_bnsd']
    topk, seq = kw['topk_idx'], kw['seq_lens']
    rtt, rows = kw.get('req_to_token'), kw.get('req_pool_indices')
    stack = _get_current_dispatch_mode_stack()
    supported = (
        _original is _ORIGINAL
        and torch.ops.npu.npu_incre_flash_attention is _IFA_OP
        and torch_npu.npu_incre_flash_attention is _IFA_OP
        and _IFA_OP.default is _IFA_DEFAULT
        and _forward_batch.forward_mode in (_backend_module.ForwardMode.DECODE,
                                            _backend_module.ForwardMode.IDLE)
        and _backend_module.get_spec().speculative_algorithm is None
        and not _backend_module._native_attn_enabled()
        and not _backend_module._native_indexer_enabled()
        and not kw['use_native'] and kw['sink'] is None
        and _idx_v_cache is None and _idx_q.shape[1:] == (1, 128)
        and _backend.page_size == kw['block_size'] == 128
        and all(type(t) is torch.Tensor for t in (q, k, v, topk, seq, rtt, rows))
        and q.device.type == 'npu' and q.dtype == torch.bfloat16
        and q.ndim == 3 and q.shape[0] == 16 and q.shape[1:] == (16, 128)
        and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        and k.shape == v.shape and k.shape[1:] == (128, 1, 128)
        and k.shape[0] >= 17 * q.shape[0] and k.numel() <= 2**31 - 1
        and k.dtype == v.dtype == q.dtype
        and topk.shape == (1, q.shape[0], 17) and topk.dtype == torch.int32
        and seq.shape == rows.shape == (q.shape[0],)
        and rtt.ndim == 2 and 128 <= rtt.shape[1] <= 2**31 - 1
        and all(t.device == q.device for t in (k, v, topk, seq, rtt, rows))
        and all(t.dtype in (torch.int32, torch.int64) for t in (seq, rtt, rows))
        and kw.get('block_table') is None
        and not _len_torch_dispatch_stack_pre_dispatch()
        and (not stack or (len(stack) == 1 and type(stack[0]) is _GraphDispatchMode))
        and not torch.compiler.is_compiling()
    )
    if not supported:
        return _original(**kw)
    capturing = torch.npu.is_current_stream_capturing()
    output = fixed_sparse_decode(q, k, v, topk, seq, rtt, rows, kw['sm_scale'])
    key = (q.device.index, q.shape[0], capturing)
    if key not in _logged:
        _logged.add(key)
        if capturing:
            logger.warning('FL_NPU_MINIMAX_DECODE_IFA device=%s rows=%s capture=%s virtual_kv=2176', *key)
    return output


def _transform(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == 'flash_decode_bnsd_with_gqa_share_sparse']
    if len(calls) != 1:
        raise RuntimeError('Unknown sparse decode attention call')
    call = calls[0]
    call.func.id = _HELPER
    call.keywords += [ast.keyword(arg=name, value=ast.Name(id=value, ctx=ast.Load()))
                      for name, value in (
                          ('_original', 'flash_decode_bnsd_with_gqa_share_sparse'),
                          ('_backend', 'self'), ('_forward_batch', 'forward_batch'),
                          ('_idx_q', 'idx_q'), ('_idx_v_cache', 'idx_v_cache'))]
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    namespace = dict(original.__globals__)
    exec(compile(tree, original.__code__.co_filename, 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    compiled = namespace[original.__name__]
    function = FunctionType(compiled.__code__, original.__globals__, original.__name__,
                            original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(function, original)
    return function


def patch_minimax_m3_decode_ifa():
    """Cold opt-in; never install into a process that already holds graphs."""
    global _ORIGINAL, _backend_module, _IFA_OP, _IFA_DEFAULT
    if os.getenv('SGLANG_FL_ASCEND_MINIMAX_DECODE_IFA', '0') != '1':
        return False
    from sgl_kernel_npu.attention import gqa_share_sparse_attention as attention
    from sglang.srt.layers.attention import minimax_sparse_backend as backend
    import torch.utils._python_dispatch as dispatch
    import torch_npu.npu.graphs as graphs

    installed = getattr(backend, '_sglang_fl_decode_ifa_binding', None)
    if installed is not None:
        if (backend.MiniMaxSparseAttnBackend._forward_npu_triton_decode is installed
                and vars(backend).get(_HELPER) is _dispatch):
            return True
        raise RuntimeError('Decode IFA bindings changed; discard old graphs')
    for name, module in [('backend', backend), ('attention', attention),
                         ('graphs', graphs), ('dispatch', dispatch)]:
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != _SOURCES[name]:
            raise RuntimeError(f'Unsupported decode IFA source: {name} {actual}')
    original = backend.MiniMaxSparseAttnBackend._forward_npu_triton_decode
    code = compile(Path(backend.__file__).read_text(), backend.__file__, 'exec', dont_inherit=True)
    class_code = next(c for c in code.co_consts if isinstance(c, CodeType)
                      and c.co_name == 'MiniMaxSparseAttnBackend')
    expected = next(c for c in class_code.co_consts if isinstance(c, CodeType)
                    and c.co_name == original.__name__)
    if (not isinstance(original, FunctionType) or original.__code__ != expected
            or original.__globals__ is not vars(backend) or original.__closure__
            or _HELPER in vars(backend)):
        raise RuntimeError('Unknown MiniMax decode binding')
    if backend._native_attn_enabled() or backend._native_indexer_enabled():
        raise RuntimeError('Keep stock native sparse switches off for fixed IFA')
    from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS
    expected_args = ('query key value padding_mask atten_mask pse_shift actual_seq_lengths '
                     'antiquant_scale antiquant_offset block_table dequant_scale1 quant_scale1 '
                     'dequant_scale2 quant_scale2 quant_offset2 kv_padding_size num_heads '
                     'scale_value input_layout num_key_value_heads block_size inner_precise').split()
    op = torch.ops.npu.npu_incre_flash_attention
    schema = op.default._schema
    if (type(op) is not torch._ops.OpOverloadPacket
            or torch_npu.npu_incre_flash_attention is not op
            or schema.name != 'npu::npu_incre_flash_attention'
            or [arg.name for arg in schema.arguments] != expected_args
            or len(schema.returns) != 1 or str(schema.returns[0].type) != 'Tensor'
            or any(name.startswith('npu_incre_flash_attention') for name in _NPU_GRAPH_OP_HANDLERS)):
        raise RuntimeError('Unknown IFA schema/alias or graph handler')
    transformed = _transform(original)
    _IFA_OP, _IFA_DEFAULT = op, op.default
    _ORIGINAL = attention.flash_decode_bnsd_with_gqa_share_sparse
    _backend_module = backend
    vars(backend)[_HELPER] = _dispatch
    backend.MiniMaxSparseAttnBackend._forward_npu_triton_decode = transformed
    backend._sglang_fl_decode_ifa_binding = transformed
    logger.warning('Enabled FL MiniMax fixed-metadata decode IFA (17 pages, graph fences preserved)')
    return True
