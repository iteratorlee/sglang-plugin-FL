# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Fuse single-head candidate merge and first-occurrence block deduplication."""

import torch
import triton
import triton.language as tl

_CONTRACT = "_sglang_fl_minimax_vector_candidates"


def mark_vector_candidates(scores, indices, seq_lens, stride_sl_b, stride_sl_h):
    """Bind the two tensors to the vector producer's ordered short-page contract."""
    marker = (id(scores), id(indices), id(seq_lens), stride_sl_b, stride_sl_h)
    setattr(scores, _CONTRACT, marker)
    setattr(indices, _CONTRACT, marker)


def clear_vector_candidates(scores, indices):
    for tensor in (scores, indices):
        if hasattr(tensor, _CONTRACT):
            delattr(tensor, _CONTRACT)


@triton.jit
def _merge_unique_kernel(
    scores_ptr,
    indices_ptr,
    seq_ptr,
    out_ptr,
    num_blocks,
    stride_sl_b,
    stride_sl_h,
    cs_c,
    cs_h,
    cs_b,
    cs_t,
    ci_c,
    ci_h,
    ci_b,
    ci_t,
    out_h,
    out_b,
    out_t,
    CHUNKS: tl.constexpr,
    CANDIDATES: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    OUT_TILE: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    length = tl.load(seq_ptr + batch * stride_sl_b + head * stride_sl_h).to(tl.int32)
    out_base = out_ptr + head * out_h + batch * out_b
    lanes = tl.arange(0, OUT_TILE)
    blocks = tl.cdiv(length, PAGE)
    # Only the marked vector producer guarantees these short candidates.
    # Zero lengths retain the original local-block convention below.
    if (length > 0) & (blocks <= TOPK) & (blocks <= num_blocks):
        unique = tl.where(lanes < blocks, lanes, -1).to(tl.int32)
        tl.store(out_base + lanes * out_t, unique, mask=lanes < TOPK + 1)
        return

    positions = tl.arange(0, CANDIDATES)
    chunk = positions // TOPK
    rank_in_chunk = positions % TOPK
    valid_candidate = positions < CHUNKS * TOPK
    scores = tl.load(
        scores_ptr + chunk * cs_c + head * cs_h + batch * cs_b + rank_in_chunk * cs_t,
        mask=valid_candidate,
        other=float("-inf"),
    ).to(tl.float32)
    indices = tl.load(
        indices_ptr + chunk * ci_c + head * ci_h + batch * ci_b + rank_in_chunk * ci_t,
        mask=valid_candidate,
        other=-1,
    ).to(tl.int32)
    scores = tl.where(indices >= 0, scores, float("-inf"))
    query_pos = tl.maximum(length - 1, 0)
    local_block = tl.minimum(query_pos // PAGE, num_blocks - 1)
    selected_raw = tl.full((OUT_TILE,), -1, tl.int32)
    unique = tl.full((OUT_TILE,), -1, tl.int32)
    local_hit = 0
    for rank in tl.static_range(TOPK):
        best_score = tl.max(scores, axis=0)
        # Keep upstream candidate-position tie order, including -inf ties.
        best_position = tl.min(
            tl.where(scores == best_score, positions, CANDIDATES), axis=0
        )
        selected = tl.max(tl.where(positions == best_position, indices, -1), axis=0)
        valid = (
            (selected >= 0) & (selected < num_blocks) & (selected * PAGE <= query_pos)
        )
        value = tl.where(valid, selected, -1)
        duplicate = (
            tl.sum(((lanes < rank) & (selected_raw == value)).to(tl.int32), axis=0) > 0
        )
        selected_raw = tl.where(lanes == rank, value, selected_raw)
        unique = tl.where(lanes == rank, tl.where(duplicate, -1, value), unique)
        local_hit += (valid & (selected == local_block)).to(tl.int32)
        scores = tl.where(positions == best_position, float("-inf"), scores)
    local_value = tl.where(local_hit > 0, -1, local_block)
    duplicate_local = (
        tl.sum(((lanes < TOPK) & (selected_raw == local_value)).to(tl.int32), axis=0)
        > 0
    )
    unique = tl.where(lanes == TOPK, tl.where(duplicate_local, -1, local_value), unique)
    tl.store(out_base + lanes * out_t, unique, mask=lanes < TOPK + 1)


def merge_vector_candidates(
    candidate_scores,
    candidate_indices,
    seq_lens,
    block_size,
    topk,
    stride_sl_b,
    stride_sl_h,
    num_blocks,
):
    """Return None for unsupported/unmarked producers; never inspect GPU values."""
    marker = (
        id(candidate_scores),
        id(candidate_indices),
        id(seq_lens),
        stride_sl_b,
        stride_sl_h,
    )
    if not (
        getattr(candidate_scores, _CONTRACT, None) == marker
        and getattr(candidate_indices, _CONTRACT, None) == marker
        and candidate_scores.device.type == "npu"
        and candidate_scores.dtype == torch.float32
        and candidate_indices.dtype == torch.int32
        and candidate_scores.shape == candidate_indices.shape
        and candidate_scores.ndim == 4
        and candidate_scores.shape[1] == 1
        and candidate_scores.shape[3] == topk == 16
        and block_size == 128
        and seq_lens.dtype in (torch.int32, torch.int64)
        and candidate_indices.device == candidate_scores.device == seq_lens.device
    ):
        return None
    chunks, heads, batch, _ = candidate_scores.shape
    if chunks < 1 or batch < 1:
        return None
    out = torch.empty(
        (heads, batch, topk + 1), dtype=torch.int32, device=candidate_scores.device
    )
    _merge_unique_kernel[(batch, heads)](
        candidate_scores,
        candidate_indices,
        seq_lens,
        out,
        num_blocks,
        stride_sl_b,
        stride_sl_h,
        *candidate_scores.stride(),
        *candidate_indices.stride(),
        *out.stride(),
        CHUNKS=chunks,
        CANDIDATES=triton.next_power_of_2(chunks * topk),
        TOPK=topk,
        PAGE=block_size,
        OUT_TILE=triton.next_power_of_2(topk + 1),
        num_warps=1,
        num_stages=1,
    )
    return out
