# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
"""Vector-only MiniMax single-head decode block scoring on Ascend."""

import logging
import os

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fl_minimax_headone_vector_score_kernel(
    q_ptr,
    k_ptr,
    page_ptr,
    req_ids,
    scores_ptr,
    indices_ptr,
    seq_ptr,
    batch_size,
    chunks,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_d,
    stride_sl_b,
    stride_page_r,
    stride_page_t,
    max_cols,
    num_pages,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    DIRECT: tl.constexpr,
    SANITIZE: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    TOPK: tl.constexpr,
    FILL: tl.constexpr,
    RUNTIME_FILL: tl.constexpr,
):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    batch = pid % batch_size
    chunk = pid // batch_size
    length = tl.load(seq_ptr + batch * stride_sl_b).to(tl.int32)
    blocks = tl.cdiv(length, PAGE)
    lanes = tl.arange(0, TOPK)
    cs = chunk * stride_cs_c + head * stride_cs_h + batch * stride_cs_b
    ci = chunk * stride_ci_c + head * stride_ci_h + batch * stride_ci_b
    top_scores = tl.full((TOPK,), float("-inf"), tl.float32)
    top_indices = tl.full((TOPK,), -1, tl.int32)

    if blocks <= TOPK:
        valid = (chunk == 0) & (lanes < blocks)
        top_scores = tl.where(valid, 0.0, top_scores)
        top_indices = tl.where(valid, lanes, top_indices)
        tl.store(scores_ptr + cs + lanes * stride_cs_t, top_scores)
        tl.store(indices_ptr + ci + lanes * stride_ci_t, top_indices)
        return

    chunk_size = tl.maximum(1, tl.cdiv(blocks, chunks))
    begin = chunk * chunk_size
    end = tl.minimum(begin + chunk_size, blocks)
    if begin >= end:
        tl.store(scores_ptr + cs + lanes * stride_cs_t, top_scores)
        tl.store(indices_ptr + ci + lanes * stride_ci_t, top_indices)
        return

    dims = tl.arange(0, D)
    tokens = tl.arange(0, PAGE)
    query = tl.load(
        q_ptr + batch * stride_q_b + head * stride_q_h + dims * stride_q_d
    ).to(tl.float32)
    if DIRECT:
        request = tl.load(req_ids + batch).to(tl.int64)
    for step in tl.range(end - begin):
        logical = begin + step
        if DIRECT:
            col = tl.minimum(logical * PAGE, max_cols - 1)
            token_slot = tl.load(
                page_ptr + request * stride_page_r + col * stride_page_t
            ).to(tl.int64)
            physical = token_slot // PAGE
            if SANITIZE:
                physical = tl.minimum(tl.maximum(physical, 0), num_pages - 1)
        else:
            physical = tl.load(
                page_ptr + batch * stride_page_r + logical * stride_page_t
            ).to(tl.int64)
        positions = logical * PAGE + tokens
        keys = tl.load(
            k_ptr
            + physical * stride_k_page
            + tokens[:, None] * stride_k_token
            + head * stride_k_head
            + dims[None, :] * stride_k_d,
            mask=positions[:, None] < length,
            other=0.0,
        ).to(tl.float32)
        token_scores = tl.sum(keys * query[None, :], axis=1)
        token_scores = token_scores * (sm_scale * 1.4426950409)
        score = tl.max(
            tl.where(positions < length, token_scores, float("-inf")), axis=0
        )
        if FILL or (RUNTIME_FILL and chunk_size <= TOPK):
            replace = lanes == step
        else:
            minimum = tl.min(top_scores, axis=0)
            minimum_lane = tl.min(tl.where(top_scores == minimum, lanes, TOPK), axis=0)
            replace = (lanes == minimum_lane) & (score > minimum)
        top_scores = tl.where(replace, score, top_scores)
        top_indices = tl.where(replace, logical, top_indices)
    tl.store(scores_ptr + cs + lanes * stride_cs_t, top_scores)
    tl.store(indices_ptr + ci + lanes * stride_ci_t, top_indices)


_ARG_NAMES = (
    "q",
    "k",
    "bt",
    "rtt",
    "req_ids",
    "cs",
    "ci",
    "seq",
    "sl_b",
    "sl_h",
    "batch",
    "gqa",
    "dim",
    "page",
    "scale",
    "init",
    "local",
    "chunks",
    "q_b",
    "q_h",
    "q_d",
    "k_p",
    "k_t",
    "k_h",
    "k_d",
    "bt_b",
    "bt_n",
    "rtt_r",
    "rtt_t",
    "max_cols",
    "pages",
    "cs_c",
    "cs_h",
    "cs_b",
    "cs_t",
    "ci_c",
    "ci_h",
    "ci_b",
    "ci_t",
)


@triton.jit
def _fl_minimax_deduplicate_blocks(
    indices, stride_h, stride_b, stride_t, SIZE: tl.constexpr, TILE: tl.constexpr
):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    lanes = tl.arange(0, TILE)
    base = indices + head * stride_h + batch * stride_b
    original = tl.load(base + lanes * stride_t, mask=lanes < SIZE, other=-1)
    for rank in tl.static_range(SIZE):
        block = tl.sum(tl.where(lanes == rank, original, 0), axis=0)
        duplicate = tl.sum(((lanes < rank) & (original == block)).to(tl.int32), axis=0)
        tl.store(base + rank * stride_t, tl.where(duplicate > 0, -1, block))


def _patch_single_head_merge(indexer):
    original = indexer._merge_topk_append_local
    if getattr(original, "_sglang_fl_unique_blocks", False):
        return

    def merge(*args, **kwargs):
        if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_FUSED_MERGE") == "1":
            from .minimax_m3_indexer_merge import merge_vector_candidates

            fused = merge_vector_candidates(*args, **kwargs)
            if fused is not None:
                return fused
        result = original(*args, **kwargs)
        if result.shape[0] == 1:
            _fl_minimax_deduplicate_blocks[(result.shape[0], result.shape[1])](
                result,
                *result.stride(),
                SIZE=result.shape[2],
                TILE=triton.next_power_of_2(result.shape[2]),
                num_warps=4,
                num_stages=1,
            )
        return result

    merge._sglang_fl_unique_blocks = True
    indexer._merge_topk_append_local = merge


class _VectorScoreDispatch:
    _sglang_fl_vector_score = True
    # For the guarded head-one contract all chunks write every TOPK slot.
    _sglang_fl_all_candidates_written = True

    def __init__(self, original):
        self.original = original
        self.logged = False

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if len(args) != len(_ARG_NAMES):
                if (
                    os.environ.get("SGLANG_FL_ASCEND_MINIMAX_FUSED_MERGE") == "1"
                    and len(args) >= 7
                ):
                    from .minimax_m3_indexer_merge import clear_vector_candidates

                    # Current ABI output slots; clearing only our private tags
                    # cannot affect an unsupported producer's tensor values.
                    clear_vector_candidates(args[5], args[6])
                return self.original[grid](*args, **kwargs)
            p = dict(zip(_ARG_NAMES, args))
            fused_merge = os.environ.get("SGLANG_FL_ASCEND_MINIMAX_FUSED_MERGE") == "1"
            if fused_merge:
                from .minimax_m3_indexer_merge import (
                    clear_vector_candidates,
                    mark_vector_candidates,
                )

                clear_vector_candidates(p["cs"], p["ci"])
            if not (
                p["gqa"] == 1
                and p["dim"] == 128
                and p["page"] == 128
                and p["init"] == 0
                and p["local"] == 0
                and kwargs.get("topk") == 16
                and kwargs.get("SCORE_TYPE") == "max"
                and not kwargs.get("RUNTIME_ADAPTIVE_SCORE_CHUNKS", False)
            ):
                return self.original[grid](*args, **kwargs)
            direct = kwargs["USE_DIRECT_PAGE_LOOKUP"]
            if fused_merge:
                mark_vector_candidates(p["cs"], p["ci"], p["seq"], p["sl_b"], p["sl_h"])
            if not self.logged:
                logger.warning("Using FL vector-only MiniMax single-head indexer")
                self.logged = True
            return _fl_minimax_headone_vector_score_kernel[grid](
                p["q"],
                p["k"],
                p["rtt"] if direct else p["bt"],
                p["req_ids"],
                p["cs"],
                p["ci"],
                p["seq"],
                p["batch"],
                p["chunks"],
                p["scale"],
                p["q_b"],
                p["q_h"],
                p["q_d"],
                p["k_p"],
                p["k_t"],
                p["k_h"],
                p["k_d"],
                p["sl_b"],
                p["rtt_r"] if direct else p["bt_b"],
                p["rtt_t"] if direct else p["bt_n"],
                p["max_cols"],
                p["pages"],
                p["cs_c"],
                p["cs_h"],
                p["cs_b"],
                p["cs_t"],
                p["ci_c"],
                p["ci_h"],
                p["ci_b"],
                p["ci_t"],
                DIRECT=direct,
                SANITIZE=kwargs["SANITIZE_PAGE_IDS"],
                D=128,
                PAGE=128,
                TOPK=16,
                FILL=kwargs["FILL_ONLY"],
                RUNTIME_FILL=kwargs["RUNTIME_FILL_ONLY"],
                num_warps=4,
                num_stages=1,
            )

        return launch


def patch_ascend_minimax_vector_indexer() -> None:
    if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_VECTOR_INDEXER") != "1":
        return
    import sgl_kernel_npu.indexer.flash_block_score_decode as indexer

    original = indexer._decode_bnsd_score_topk_chunk_kernel
    if getattr(original, "_sglang_fl_vector_score", False):
        return
    indexer._decode_bnsd_score_topk_chunk_kernel = _VectorScoreDispatch(original)
    _patch_single_head_merge(indexer)
    logger.warning("Enabled FL vector-only MiniMax indexer dispatch")
