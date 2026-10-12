# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
"""Elide candidate fills only for the verified fully writing Vector producer.

The score-only body follows the pinned upstream function; its arithmetic,
chunk selection and merge are unchanged. Unknown source/ABI uses upstream.
"""

import functools
import hashlib
import importlib
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)

_UPSTREAM_FUNCTION_SHA256 = (
    "69f6d42f74f94ceb3253e5b1c114c5055c9d7f9fde95a728713f638c30f359bb"
)
_EXPECTED_PARAMETERS = frozenset(
    [
        "q",
        "sink",
        "k_cache_bnsd",
        "v_cache_bnsd",
        "block_table",
        "seq_lens",
        "max_seqlen",
        "block_size",
        "topk",
        "init_blocks",
        "local_blocks",
        "sm_scale",
        "score_type",
        "disable_index_value",
        "num_kv_chunks",
        "max_num_kv_chunks",
        "use_triton_topk",
        "num_topk_chunks",
        "use_chunked_score",
        "score_blocks_per_chunk",
        "score_max_chunks",
        "req_to_token",
        "req_pool_indices",
        "max_num_blocks",
        "num_pages",
        "sanitize_page_ids",
        "packed_seq_lens",
        "runtime_fill_only",
        "runtime_score_short_max_blocks",
        "runtime_score_short_chunks",
        "fused_append_local",
        "use_native",
    ]
)


def _score_with_empty(indexer, p):
    q = p["q"]
    k_cache_bnsd = p["k_cache_bnsd"]
    seq_lens = p["seq_lens"]
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = 1
    gqa_group_size = 1
    stride_sl_b, stride_sl_h = 1, 0
    use_direct_page_lookup = True
    req_to_token = p["req_to_token"]
    req_pool_indices = p["req_pool_indices"]
    page_source, page_source_rows = req_to_token, req_pool_indices
    direct_num_pages = int(p["num_pages"])
    max_num_blocks = p["max_num_blocks"]
    block_size, topk = p["block_size"], p["topk"]
    init_blocks, local_blocks = p["init_blocks"], p["local_blocks"]
    sm_scale = p["sm_scale"] if p["sm_scale"] is not None else head_dim**-0.5
    score_type = p["score_type"]
    score_blocks_per_chunk, score_max_chunks = (
        p["score_blocks_per_chunk"],
        p["score_max_chunks"],
    )
    runtime_score_short_chunks = p["runtime_score_short_chunks"]
    runtime_score_short_max_blocks = p["runtime_score_short_max_blocks"]
    runtime_fill_only = p["runtime_fill_only"]
    fused_append_local = p["fused_append_local"]
    sanitize_page_ids = p["sanitize_page_ids"]
    max_seqblock = (p["max_seqlen"] + block_size - 1) // block_size
    block_size_n = 128
    if topk <= 0:
        return (
            None,
            torch.empty(
                (num_q_heads, batch_size, 0), dtype=torch.int32, device=q.device
            ),
        )
    num_score_chunks = indexer._choose_num_score_chunks(
        max_seqblock,
        blocks_per_chunk=score_blocks_per_chunk,
        max_chunks=score_max_chunks,
        all_seqblock_q=batch_size,
        num_kv_heads=num_kv_heads,
    )
    if runtime_score_short_chunks:
        runtime_score_short_chunks = min(runtime_score_short_chunks, num_score_chunks)
    use_runtime_adaptive_score_chunks = bool(
        runtime_score_short_max_blocks or runtime_score_short_chunks
    )
    if use_runtime_adaptive_score_chunks:
        assert runtime_score_short_max_blocks > 0
        assert runtime_score_short_chunks > 0
        assert runtime_score_short_chunks <= num_score_chunks
        assert runtime_score_short_chunks & runtime_score_short_chunks - 1 == 0
    candidate_scores = torch.empty(
        (num_score_chunks, num_q_heads, batch_size, topk),
        dtype=torch.float32,
        device=q.device,
    )
    candidate_indices = torch.empty(
        (num_score_chunks, num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=q.device,
    )
    indexer._decode_bnsd_score_topk_chunk_kernel[
        batch_size * num_score_chunks, num_kv_heads
    ](
        q,
        k_cache_bnsd,
        page_source,
        page_source,
        page_source_rows,
        candidate_scores,
        candidate_indices,
        seq_lens,
        stride_sl_b,
        stride_sl_h,
        batch_size,
        gqa_group_size,
        head_dim,
        block_size,
        sm_scale,
        init_blocks,
        local_blocks,
        num_score_chunks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        page_source.stride(0),
        page_source.stride(1),
        req_to_token.stride(0) if use_direct_page_lookup else 0,
        req_to_token.stride(1) if use_direct_page_lookup else 0,
        req_to_token.shape[1] if use_direct_page_lookup else 1,
        direct_num_pages,
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_scores.stride(2),
        candidate_scores.stride(3),
        candidate_indices.stride(0),
        candidate_indices.stride(1),
        candidate_indices.stride(2),
        candidate_indices.stride(3),
        BLOCK_SIZE_N=block_size_n,
        SCORE_TYPE=score_type,
        topk=topk,
        USE_DIRECT_PAGE_LOOKUP=use_direct_page_lookup,
        SANITIZE_PAGE_IDS=sanitize_page_ids,
        FILL_ONLY=(max_seqblock + num_score_chunks - 1) // num_score_chunks <= topk,
        RUNTIME_FILL_ONLY=runtime_fill_only,
        RUNTIME_ADAPTIVE_SCORE_CHUNKS=use_runtime_adaptive_score_chunks,
        RUNTIME_SCORE_SHORT_MAX_BLOCKS=runtime_score_short_max_blocks,
        RUNTIME_SCORE_SHORT_CHUNKS=runtime_score_short_chunks,
        num_warps=indexer._SCORE_CHUNK_NW,
        num_stages=indexer._SCORE_CHUNK_NS,
    )
    if fused_append_local:
        assert max_num_blocks is not None and max_num_blocks > 0
        if use_runtime_adaptive_score_chunks:
            topk_indices = indexer._merge_topk_append_local_adaptive(
                candidate_scores,
                candidate_indices,
                seq_lens,
                block_size,
                topk,
                stride_sl_b,
                stride_sl_h,
                max_num_blocks,
                runtime_score_short_max_blocks,
                runtime_score_short_chunks,
            )
        else:
            topk_indices = indexer._merge_topk_append_local(
                candidate_scores,
                candidate_indices,
                seq_lens,
                block_size,
                topk,
                stride_sl_b,
                stride_sl_h,
                max_num_blocks,
            )
    elif use_runtime_adaptive_score_chunks:
        topk_indices = indexer._merge_bnsd_score_topk_candidates_adaptive(
            candidate_scores,
            candidate_indices,
            seq_lens,
            block_size,
            topk,
            stride_sl_b,
            stride_sl_h,
            runtime_score_short_max_blocks,
            runtime_score_short_chunks,
        )
    else:
        topk_indices = indexer._merge_bnsd_score_topk_candidates(
            candidate_scores, candidate_indices, topk
        )
    return (None, topk_indices)


def _supported(indexer, p):
    producer = indexer._decode_bnsd_score_topk_chunk_kernel
    if not getattr(producer, "_sglang_fl_all_candidates_written", False):
        return False
    q, k = p["q"], p["k_cache_bnsd"]
    seq, table, requests = p["seq_lens"], p["req_to_token"], p["req_pool_indices"]
    if not all(isinstance(t, torch.Tensor) for t in (q, k, seq, table, requests)):
        return False
    return (
        q.device.type == "npu"
        and all(t.device == q.device for t in (k, seq, table, requests))
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and q.ndim == 3
        and q.shape[0] > 0
        and q.shape[1:] == (1, 128)
        and k.ndim == 4
        and k.shape[1:] == (128, 1, 128)
        and seq.ndim == 1
        and seq.shape[0] == q.shape[0]
        and seq.is_contiguous()
        and seq.dtype in (torch.int32, torch.int64)
        and table.ndim == 2
        and table.dtype in (torch.int32, torch.int64)
        and requests.ndim == 1
        and requests.shape[0] == q.shape[0]
        and requests.is_contiguous()
        and requests.dtype in (torch.int32, torch.int64)
        and p["sink"] is None
        and p["block_table"] is None
        and p["disable_index_value"]
        and p["block_size"] == 128
        and p["topk"] == 16
        and p["init_blocks"] == 0
        and p["local_blocks"] == 0
        and p["score_type"] == "max"
        and not p["packed_seq_lens"]
        and not p["runtime_score_short_max_blocks"]
        and not p["runtime_score_short_chunks"]
        and p["fused_append_local"]
        and isinstance(p["max_num_blocks"], int)
        and p["max_num_blocks"] > 0
        and p["max_num_blocks"] * 128 <= table.shape[1]
        and isinstance(p["num_pages"], int)
        and 0 < p["num_pages"] <= k.shape[0]
        and p["max_seqlen"] > 0
        and p["score_blocks_per_chunk"] > 0
        and p["score_max_chunks"] > 0
    )


def patch_ascend_minimax_score_buffers():
    if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_ELIDE_SCORE_FILLS") != "1":
        return
    indexer = importlib.import_module("sgl_kernel_npu.indexer.flash_block_score_decode")
    original = indexer.flash_decode_bnsd_with_topk_idx
    if getattr(original, "_sglang_fl_empty_score_buffers", False):
        return
    try:
        signature = inspect.signature(original)
        source_hash = hashlib.sha256(
            inspect.getsource(original).strip().encode()
        ).hexdigest()
    except (OSError, TypeError, ValueError):
        logger.warning("MiniMax empty score buffers disabled: source/ABI unavailable")
        return
    if (
        set(signature.parameters) != _EXPECTED_PARAMETERS
        or source_hash != _UPSTREAM_FUNCTION_SHA256
    ):
        logger.warning(
            "MiniMax empty score buffers disabled: upstream source/ABI changed"
        )
        return
    logged = False

    @functools.wraps(original)
    @torch.no_grad()
    def score(*args, **kwargs):
        nonlocal logged
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not _supported(indexer, bound.arguments):
            return original(*args, **kwargs)
        if not logged:
            logger.warning(
                "Using FL fully written MiniMax score buffers without initialization"
            )
            logged = True
        return _score_with_empty(indexer, bound.arguments)

    score._sglang_fl_empty_score_buffers = True
    score._sglang_fl_empty_score_original = original
    indexer.flash_decode_bnsd_with_topk_idx = score
    # MiniMax decode imports this function locally on each call.
    logger.warning("Enabled FL guarded MiniMax score-buffer fill elimination")
