# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Skip entirely invisible KV pages after the pinned prefill causal fix.

The original score slab remains initialized to -inf; this is required for
all skipped pages. Only BPS1/max dispatch selects the optimized kernel.
"""

import hashlib
import inspect
import logging
import os
from pathlib import Path

import triton
import triton.language as tl

from . import minimax_m3_prefill_causal as causal

logger = logging.getLogger(__name__)
_CAUSAL_KERNEL_SHA256 = (
    "8f5c8a6993dbe4c6e60055afb86f77cce5ccf68ea6a2c2a78e4bd413fdfe24a4"
)


@triton.jit
def _fl_prefill_future_score_kernel(
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
    q_start = tl.load(qb_to_qstart_ptr + pid_qb).to(tl.int32)
    q_block_local = tl.load(qb_to_qblock_ptr + pid_qb).to(tl.int32)
    q_end = tl.load(qb_qend_ptr + pid_qb).to(tl.int32)
    if BLOCKS_PER_STEP == 1:
        last_local = tl.minimum((q_block_local + 1) * BLOCK_SIZE_Q, q_end - q_start) - 1
        visible_blocks = tl.cdiv(
            seq_len - (q_end - q_start) + last_local + 1, block_size
        )
        chunk_end_block = tl.minimum(chunk_end_block, visible_blocks)
    if chunk_start_block >= chunk_end_block:
        return

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


class _FutureKernelDispatch:
    _sglang_fl_prefill_future = True

    def __init__(self, original):
        self.original = original
        self.fn = original.fn
        self.logged = False
        parameters = list(inspect.signature(original.fn).parameters)
        self.score_type_position = parameters.index("SCORE_TYPE")
        self.blocks_per_step_position = parameters.index("BLOCKS_PER_STEP")

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            score_type = kwargs.get(
                "SCORE_TYPE",
                args[self.score_type_position]
                if len(args) > self.score_type_position
                else None,
            )
            blocks_per_step = kwargs.get(
                "BLOCKS_PER_STEP",
                args[self.blocks_per_step_position]
                if len(args) > self.blocks_per_step_position
                else None,
            )
            if blocks_per_step != 1 or score_type != "max":
                return self.original[grid](*args, **kwargs)
            if not self.logged:
                logger.warning("Using FL MiniMax prefill whole-future-page skipping")
                self.logged = True
            return _fl_prefill_future_score_kernel[grid](*args, **kwargs)

        return launch


def patch_ascend_minimax_prefill_future():
    if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_PREFILL_FUTURE_SKIP") != "1":
        return
    if getattr(
        causal._fl_prefill_bnsd_score_kernel, "_sglang_fl_prefill_future", False
    ):
        return
    try:
        from sgl_kernel_npu.indexer import flash_block_score_prefill as module
        from sglang.srt.layers.attention import minimax_sparse_backend as backend

        proxy = module._prefill_bnsd_score_kernel
        original = causal._fl_prefill_bnsd_score_kernel
        compatible = (
            type(proxy) is causal._PrefillCausalDispatch
            and hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            == causal._MODULE_SHA256
            and hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest()
            == causal._BACKEND_SHA256
            and hashlib.sha256(inspect.getsource(original.fn).encode()).hexdigest()
            == _CAUSAL_KERNEL_SHA256
            and inspect.signature(original.fn)
            == inspect.signature(_fl_prefill_future_score_kernel.fn)
        )
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        logger.warning(
            "MiniMax prefill future skipping disabled: ABI/source check failed: %s", exc
        )
        return
    if not compatible:
        logger.warning(
            "MiniMax prefill future skipping disabled: unknown causal proxy/ABI/source"
        )
        return
    causal._fl_prefill_bnsd_score_kernel = _FutureKernelDispatch(original)
    logger.warning("Enabled FL MiniMax prefill whole-future-page skipping")
