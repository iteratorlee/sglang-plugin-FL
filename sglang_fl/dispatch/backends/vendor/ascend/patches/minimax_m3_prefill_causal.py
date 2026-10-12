# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Correct request-local causality in the pinned MiniMax prefill score kernel.

Adapted from sgl_kernel_npu 2026.9.0. The score-only kernel's arithmetic,
tiling and packed input/output addresses are preserved. The index-V fused
attention kernel is not replaced; the current M3.1 model disables index V.
"""

import hashlib
import inspect
import logging
import os
from pathlib import Path

import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_MODULE_SHA256 = "3b01743ccbe15ad6ceb92817dff10fc4c719490ac57449b55416196b1339e2cf"
_KERNEL_SHA256 = "bfe3a911b3cc45d8c49dd50d129d397d6ebdd41aa8dd42059f579b621d971812"
_BACKEND_SHA256 = "2798456cbe4f907fbbccf1eaab1114d2324e79bf1205adb6d4c2e9b8b738b048"


@triton.jit
def _fl_prefill_bnsd_score_kernel(
    q_ptr,  # [total_q, num_idx_heads, head_dim]
    k_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim]
    block_table_ptr,  # [all_seqblock_q, max_num_blocks]
    qb_to_qstart_ptr,  # [all_seqblock_q]
    qb_to_qblock_ptr,  # [all_seqblock_q]
    qb_seq_lens_ptr,  # [all_seqblock_q]
    qb_qend_ptr,  # [all_seqblock_q] exclusive q upper bound (cu_seqlens[r+1])
    score_ptr,  # [num_idx_heads, total_q, max_seqblock_k]
    # scalars
    total_q,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    all_seqblock_q,
    num_score_chunks,
    sm_scale,
    # strides
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_bt_q,
    stride_bt_n,
    stride_s_h,
    stride_s_q,
    stride_s_n,
    # constexpr meta
    block_size: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    # Consecutive KV blocks fused into one tl.dot per step (power of two).
    BLOCKS_PER_STEP: tl.constexpr,
):
    """Score one query-block x one kv_head tile."""
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_qbc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_qb = pid_qbc % all_seqblock_q
    pid_c = pid_qbc // all_seqblock_q

    seq_len = tl.load(qb_seq_lens_ptr + pid_qb).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)
    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, num_score_chunks))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return

    q_start = tl.load(qb_to_qstart_ptr + pid_qb).to(tl.int32)
    q_block_local = tl.load(qb_to_qblock_ptr + pid_qb).to(tl.int32)
    q_end = tl.load(qb_qend_ptr + pid_qb).to(tl.int32)

    off_d = tl.arange(0, BLOCK_SIZE_D)  # [D]
    off_n = tl.arange(0, BLOCK_SIZE_N)  # [N]
    # Flatten (query-row, head-within-group) to 1D: small-H 2D fails Ascend stride alignment.
    off_qh = tl.arange(0, BLOCK_SIZE_Q * BLOCK_SIZE_H)  # [BSQ*H]
    qi = off_qh // BLOCK_SIZE_H  # query row within the block
    hh = off_qh % BLOCK_SIZE_H  # head index within the GQA group
    pid_h_base = pid_kh * gqa_group_size
    q_token_raw = q_start + q_block_local * BLOCK_SIZE_Q + qi  # [BSQ*H]
    head_flat = pid_h_base + hh  # actual idx-head index per row
    # Clamp q_token to [q_start, q_end-1] so phantom tail rows stay in-request.
    row_valid = (q_token_raw < q_end) & (hh < gqa_group_size)
    q_token_flat = tl.maximum(q_start, tl.minimum(q_token_raw, q_end - 1))

    # Packed Q addresses differ from request-local causal positions.
    request_query_pos = seq_len - (q_end - q_start) + (q_token_raw - q_start)

    # Q load: [BSQ*H, D]
    q_offsets = (
        q_token_flat[:, None] * stride_q_n
        + head_flat[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409

    if BLOCKS_PER_STEP == 1:
        # Single-block path: one KV block (block_size tokens) per dot.
        num_steps = chunk_end_block - chunk_start_block
        for step in tl.range(num_steps):
            logical_block = chunk_start_block + step
            physical_block = tl.load(
                block_table_ptr + pid_qb * stride_bt_q + logical_block * stride_bt_n
            ).to(tl.int64)

            key_pos = logical_block * block_size + off_n  # [N]
            pos_mask = key_pos < seq_len

            k_offsets = (
                physical_block * stride_k_block
                + off_n[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
                other=0.0,
            )

            qk = tl.dot(q, k) * sm_scale_log2e  # [BSQ*H, N], 2D dot

            # Causal: query token >= key position.
            causal = request_query_pos[:, None] >= key_pos[None, :]
            qk = tl.where(causal & pos_mask[None, :], qk, float("-inf"))

            sub_max = tl.max(qk, axis=1)  # [BSQ*H]
            if SCORE_TYPE == "max":
                score = sub_max
            else:
                score = sub_max + tl.log2(
                    tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1)
                )
                score = tl.where(score != score, float("-inf"), score)

            s_offsets = (
                head_flat * stride_s_h
                + q_token_raw * stride_s_q
                + logical_block * stride_s_n
            )
            tl.store(
                score_ptr + s_offsets,
                score.to(score_ptr.dtype.element_ty),
                mask=row_valid,
            )
    else:
        # Multi-block path: BLOCKS_PER_STEP consecutive KV blocks per tl.dot.
        sub_id = off_n // block_size  # [BLOCK_SIZE_N] which sub-block in [0, BPS)
        inn = off_n % block_size  # [BLOCK_SIZE_N] offset within the block
        num_steps = tl.cdiv(chunk_end_block - chunk_start_block, BLOCKS_PER_STEP)
        for step in tl.range(num_steps):
            logical_base = chunk_start_block + step * BLOCKS_PER_STEP
            logical_block_n = logical_base + sub_id  # [BLOCK_SIZE_N]
            # Clamp the block-table index to a valid block (out-of-range masked below).
            logical_block_load = tl.minimum(logical_block_n, chunk_end_block - 1)
            physical_block_n = tl.load(
                block_table_ptr
                + pid_qb * stride_bt_q
                + logical_block_load * stride_bt_n
            ).to(tl.int64)

            key_pos = logical_base * block_size + off_n  # [BLOCK_SIZE_N]
            pos_mask = key_pos < seq_len

            k_offsets = (
                physical_block_n[None, :] * stride_k_block
                + inn[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
                other=0.0,
            )

            qk = tl.dot(q, k) * sm_scale_log2e  # [BSQ*H, BLOCKS_PER_STEP*block_size]

            causal = request_query_pos[:, None] >= key_pos[None, :]
            qk = tl.where(causal & pos_mask[None, :], qk, float("-inf"))

            # Reduce + store one score per logical block.
            for j in tl.static_range(BLOCKS_PER_STEP):
                logical_block_j = logical_base + j
                col_lo = j * block_size
                col_sel = (off_n >= col_lo) & (off_n < col_lo + block_size)
                qk_j = tl.where(col_sel[None, :], qk, float("-inf"))
                sub_max_j = tl.max(qk_j, axis=1)  # [BSQ*H]
                if SCORE_TYPE == "max":
                    score_j = sub_max_j
                else:
                    score_j = sub_max_j + tl.log2(
                        tl.sum(tl.exp2(qk_j - sub_max_j[:, None]), axis=1)
                    )
                    score_j = tl.where(score_j != score_j, float("-inf"), score_j)
                s_offsets_j = (
                    head_flat * stride_s_h
                    + q_token_raw * stride_s_q
                    + logical_block_j * stride_s_n
                )
                tl.store(
                    score_ptr + s_offsets_j,
                    score_j.to(score_ptr.dtype.element_ty),
                    mask=row_valid & (logical_block_j < chunk_end_block),
                )


class _PrefillCausalDispatch:
    _sglang_fl_prefill_causal = True

    def __init__(self, original):
        self._sglang_fl_prefill_causal_original = original
        self.logged = False

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if not self.logged:
                logger.warning("Using FL MiniMax prefill score request-local causality")
                self.logged = True
            return _fl_prefill_bnsd_score_kernel[grid](*args, **kwargs)

        return launch


def patch_ascend_minimax_prefill_causal():
    if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_PREFILL_CAUSAL") != "1":
        return
    try:
        from sgl_kernel_npu.indexer import flash_block_score_prefill as module
        from sglang.srt.layers.attention import minimax_sparse_backend as backend_module

        original = module._prefill_bnsd_score_kernel
        if getattr(original, "_sglang_fl_prefill_causal", False):
            return
        backend = Path(backend_module.__file__)
        compatible = (
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            == _MODULE_SHA256
            and hashlib.sha256(inspect.getsource(original.fn).encode()).hexdigest()
            == _KERNEL_SHA256
            and hashlib.sha256(backend.read_bytes()).hexdigest() == _BACKEND_SHA256
            and inspect.signature(original.fn)
            == inspect.signature(_fl_prefill_bnsd_score_kernel.fn)
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
        StopIteration,
    ) as exc:
        logger.warning(
            "MiniMax prefill causal patch disabled: ABI/source check failed: %s", exc
        )
        return
    if not compatible:
        logger.warning("MiniMax prefill causal patch disabled: unknown ABI/source")
        return
    module._prefill_bnsd_score_kernel = _PrefillCausalDispatch(original)
    logger.warning("Enabled FL MiniMax prefill score request-local causal positions")
