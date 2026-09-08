"""Sink-aware DSA over selected KV rows on Ascend, including graph replay.

The CANN build in the SGLang 0.5.11 image returns sparse softmax statistics
only with non-paged KV layouts.  Map the indexer's logical token IDs to physical
KV rows, then present the existing pool as a TND view.  Sparse mode 0 prevents
the operator from imposing a second, physical-address-based causal mask:
causality is enforced against logical positions in the mapping kernel instead.
Only the indexer's selected rows participate in attention.  No dense attention
or full-history KV gathering is performed.
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _map_sparse_indices(
    source,
    table,
    cu_q,
    kv_lens,
    target,
    source_stride,
    table_stride,
    table_width,
    POOL_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
    TOPK: tl.constexpr,
    BATCH: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    batch_ids = tl.arange(0, BATCH_BLOCK)
    ends = tl.load(cu_q + batch_ids, batch_ids < BATCH, other=2147483647)
    batch = tl.sum(((row >= ends) & (batch_ids < BATCH)).to(tl.int32), 0)
    active = batch < BATCH
    begin = tl.load(cu_q + batch - 1, active & (batch > 0), other=0)
    end = tl.load(cu_q + batch, active, other=0)
    length = tl.load(kv_lens + batch, active, other=0)
    causal_end = length - (end - begin) + (row - begin) + 1
    logical = tl.load(source + row * source_stride + offsets, offsets < TOPK, other=-1)
    page_id = logical // PAGE
    valid = active & (offsets < TOPK) & (logical >= 0)
    valid &= (logical < causal_end) & (logical < length)
    valid &= (page_id >= 0) & (page_id < table_width)
    physical_page = tl.load(table + batch * table_stride + page_id, valid, other=-1)
    physical = physical_page * PAGE + logical % PAGE
    valid &= (physical_page >= 0) & (physical >= 0) & (physical < POOL_TOKENS)
    tl.store(
        target + row * TOPK + offsets, tl.where(valid, physical, -1), offsets < TOPK
    )


@triton.jit
def _copy_query(
    source,
    target,
    STRIDE_T: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_D: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    head, dim = index // DIM, index % DIM
    value = tl.load(
        source + row * STRIDE_T + head * STRIDE_H + dim * STRIDE_D,
        index < HEADS * DIM,
        other=0,
    )
    tl.store(target + row * HEADS * DIM + index, value, index < HEADS * DIM)


@triton.jit
def _sort_sparse_indices(source, target, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(source + row * TOPK + offsets, offsets < TOPK, other=-1)
    # Ascend's vector sort accepts floating-point values only.  Physical
    # token IDs are exact in float32 throughout HY4's native 1M-token window.
    values = tl.sort(values.to(tl.float32), descending=True).to(tl.int32)
    tl.store(target + row * TOPK + offsets, values, offsets < TOPK)


@triton.jit
def _apply_sink(
    source,
    maximum,
    denominator,
    sinks,
    target,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_head = tl.program_id(0)
    head = row_head % HEADS
    offsets = tl.arange(0, BLOCK)
    m = tl.load(maximum + row_head).to(tl.float32)
    z = tl.load(denominator + row_head).to(tl.float32)
    sink = tl.load(sinks + head).to(tl.float32)
    # Attention sinks have no value vector: they contribute exp(sink) only
    # to the denominator.  Log-space sigmoid avoids overflow for large sinks.
    lse = m + tl.log(z)
    factor = 1.0 / (1.0 + tl.exp(sink - lse))
    value = tl.load(source + row_head * DIM + offsets, offsets < DIM, other=0).to(
        tl.float32
    )
    value = tl.where(z > 0, value * factor, 0.0)
    tl.store(target + row_head * DIM + offsets, value, offsets < DIM)


def _contiguous_query(query):
    if query.is_contiguous():
        return query
    result = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    _copy_query[(query.shape[0],)](
        query,
        result,
        *query.stride(),
        query.shape[1],
        query.shape[2],
        triton.next_power_of_2(query.shape[1] * query.shape[2]),
    )
    return result


def hy4_sparse_attention(
    query_nope,
    query_rope,
    key_cache,
    rope_cache,
    topk_indices,
    block_table,
    cu_seqlens_q,
    seq_lens_kv,
    sink,
    scale,
):
    """Return [T, local_heads, kv_lora_rank] using the supplied sparse IDs.

    KV pools are [pages, page_size, 1, dim]; block_table contains physical
    page IDs.  Indexer output is [T, topk] or [T, 1, topk], with -1 padding.
    cu_seqlens_q contains cumulative query lengths without an initial zero.
    All metadata is read from device tensors on every replay.
    """
    if key_cache.ndim != 4 or rope_cache.ndim != 4:
        raise ValueError("HY4 sparse attention expects paged [P,S,1,D] KV pools")
    if key_cache.shape[2] != 1 or rope_cache.shape[2] != 1:
        raise ValueError("HY4 absorbed MLA requires one latent KV head")
    if not key_cache.is_contiguous() or not rope_cache.is_contiguous():
        raise ValueError("HY4 sparse attention requires contiguous paged KV storage")
    if not topk_indices.is_contiguous():
        raise ValueError("HY4 native indexer must return contiguous sparse indices")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("HY4 sparse indices must contain integer token positions")
    tokens, heads, dim = query_nope.shape
    indices = topk_indices.reshape(tokens, -1)
    topk = indices.shape[1]
    page = key_cache.shape[1]
    pool_tokens = key_cache.shape[0] * page
    if pool_tokens >= 2**24:
        raise ValueError("HY4 sparse physical token pool must fit exact FP32 integers")
    batches = cu_seqlens_q.numel()
    mapped = torch.empty((tokens, 1, topk), dtype=torch.int32, device=query_nope.device)
    _map_sparse_indices[(tokens, triton.cdiv(topk, 256))](
        indices,
        block_table,
        cu_seqlens_q,
        seq_lens_kv,
        mapped,
        indices.stride(0),
        block_table.stride(0),
        block_table.shape[1],
        pool_tokens,
        page,
        topk,
        batches,
        triton.next_power_of_2(batches),
        256,
    )
    packed = torch.empty_like(mapped)
    _sort_sparse_indices[(tokens,)](mapped, packed, topk, triton.next_power_of_2(topk))
    # Treat the physical pool as one sequence.  Each query already carries
    # its own masked physical sparse IDs, so no cross-request rows are read.
    physical_q_len = torch.full(
        (1,), tokens, dtype=torch.int32, device=query_nope.device
    )
    physical_kv_len = torch.full(
        (1,), pool_tokens, dtype=torch.int32, device=query_nope.device
    )
    output, maximum, denominator = torch_npu.npu_sparse_flash_attention(
        query=_contiguous_query(query_nope),
        key=key_cache.view(pool_tokens, 1, dim),
        value=key_cache.view(pool_tokens, 1, dim),
        query_rope=_contiguous_query(query_rope),
        key_rope=rope_cache.view(pool_tokens, 1, rope_cache.shape[-1]),
        sparse_indices=packed,
        scale_value=scale,
        actual_seq_lengths_query=physical_q_len,
        actual_seq_lengths_kv=physical_kv_len,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    result = torch.empty(
        (tokens, heads, dim), dtype=query_nope.dtype, device=query_nope.device
    )
    _apply_sink[(tokens * heads,)](
        output,
        maximum,
        denominator,
        sink,
        result,
        heads,
        dim,
        triton.next_power_of_2(dim),
    )
    return result
