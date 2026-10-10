# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Reuse MiniMax's current-forward CPU maximum in the NPU prefill indexer."""

import inspect
import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)
_CONTRACT = "_sglang_fl_minimax_prefill_metadata"


def _score_with_metadata(
    module, q, cache, mappings, block_q, page, scale, score_type, max_blocks
):
    import torch
    import triton

    total_q, heads, dim = q.shape
    kv_heads = cache.shape[2]
    group = heads // kv_heads
    qstart, qblock, lengths, qend, table, query_blocks = mappings
    if query_blocks == 0:
        return torch.empty(
            (heads, total_q, max_blocks), dtype=torch.float32, device=q.device
        )
    chunks = module._choose_num_score_chunks(
        max_blocks, all_seqblock_q=query_blocks, num_kv_heads=kv_heads
    )
    chunks = max(1, min(chunks, max_blocks))
    raw = os.environ.get("MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP")
    blocks_per_step = int(raw) if raw else 1
    assert blocks_per_step in (1, 2, 4), (
        "MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP must be one of 1/2/4, "
        f"got {blocks_per_step!r}"
    )
    score = torch.full(
        (heads, total_q, max_blocks),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    module._prefill_bnsd_score_kernel[(query_blocks * chunks, kv_heads)](
        q,
        cache,
        table,
        qstart,
        qblock,
        lengths,
        qend,
        score,
        total_q,
        kv_heads,
        group,
        dim,
        query_blocks,
        chunks,
        scale,
        *q.stride(),
        *cache.stride(),
        *table.stride(),
        *score.stride(),
        page,
        triton.next_power_of_2(block_q),
        triton.next_power_of_2(group),
        triton.next_power_of_2(dim),
        triton.next_power_of_2(page * blocks_per_step),
        score_type,
        blocks_per_step,
        num_warps=module._SCORE_NW,
        num_stages=module._SCORE_NS,
    )
    return score


def patch_minimax_m3_prefill_metadata() -> None:
    if os.environ.get("SGLANG_FL_ASCEND_MINIMAX_PREFILL_META") != "1":
        return
    import torch
    from sgl_kernel_npu.indexer import flash_block_score_prefill as indexer
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        MiniMaxSparseAttnBackend,
    )

    original = indexer.flash_prefill_bnsd_with_topk_idx
    build = MiniMaxSparseAttnBackend._build_prefill_meta
    if getattr(original, "_sglang_fl_prefill_metadata", False):
        return
    expected = (
        "q",
        "k_cache_bnsd",
        "cu_seqlens",
        "seq_lens",
        "req_to_token",
        "req_pool_indices",
        "block_size_q",
        "block_size_k",
        "topk",
        "sm_scale",
        "score_type",
        "qblock_mappings",
        "per_query_seq_lens",
    )
    if tuple(inspect.signature(original).parameters) != expected:
        logger.warning("Skipping MiniMax prefill metadata patch: unsupported signature")
        return
    build_signature = inspect.signature(build)
    if not {"self", "seq_lens", "page_size"}.issubset(build_signature.parameters):
        logger.warning("Skipping MiniMax prefill metadata patch: unsupported builder")
        return

    @wraps(build)
    def build_metadata(self, *args, **kwargs):
        meta = build(self, *args, **kwargs)
        bound = build_signature.bind(self, *args, **kwargs).arguments
        if not all(
            hasattr(meta, name)
            for name in ("max_blocks", "block_size_q", "per_query_seq_lens")
        ):
            return meta
        seq_lens = bound["seq_lens"]
        mappings = getattr(meta, "qblock_mappings", None)
        if not isinstance(mappings, tuple) or len(mappings) != 6:
            return meta
        table = mappings[4]
        if (
            isinstance(table, torch.Tensor)
            and table.ndim == 2
            and table.shape[1] == meta.max_blocks
        ):
            # This identity is per-forward; no cross-request shape cache.
            setattr(
                table,
                _CONTRACT,
                (
                    id(mappings),
                    id(seq_lens),
                    id(meta.per_query_seq_lens),
                    meta.block_size_q,
                    bound["page_size"],
                    meta.max_blocks,
                ),
            )
        return meta

    @wraps(original)
    def score_topk(
        q,
        k_cache_bnsd,
        cu_seqlens,
        seq_lens,
        req_to_token,
        req_pool_indices,
        block_size_q,
        block_size_k,
        topk,
        sm_scale,
        score_type="max",
        qblock_mappings=None,
        per_query_seq_lens=None,
    ):
        valid_mapping = isinstance(qblock_mappings, tuple) and len(qblock_mappings) == 6
        table = qblock_mappings[4] if valid_mapping else None
        max_blocks = (
            table.shape[1] if isinstance(table, torch.Tensor) and table.ndim == 2 else 0
        )
        expected_contract = (
            id(qblock_mappings),
            id(seq_lens),
            id(per_query_seq_lens),
            block_size_q,
            block_size_k,
            max_blocks,
        )
        eligible = (
            max_blocks > 0
            and getattr(table, _CONTRACT, None) == expected_contract
            and q.device.type == "npu"
            and q.dtype == torch.bfloat16
            and q.ndim == 3
            and q.shape[0] > 0
            and q.shape[1:] == (1, 128)
            and k_cache_bnsd.ndim == 4
            and k_cache_bnsd.shape[1:] == (128, 1, 128)
            and k_cache_bnsd.dtype == torch.bfloat16
            and block_size_k == 128
            and topk == 16
            and score_type == "max"
            and isinstance(per_query_seq_lens, torch.Tensor)
            and per_query_seq_lens.dtype == torch.int32
            and per_query_seq_lens.shape == (q.shape[0],)
            and table.dtype == torch.int32
            and isinstance(qblock_mappings[5], int)
            and table.shape[0] == qblock_mappings[5]
            and all(
                isinstance(t, torch.Tensor) and t.device == q.device
                for t in qblock_mappings[:5]
            )
            and k_cache_bnsd.device
            == q.device
            == per_query_seq_lens.device
            == seq_lens.device
        )
        if not eligible:
            return original(
                q,
                k_cache_bnsd,
                cu_seqlens,
                seq_lens,
                req_to_token,
                req_pool_indices,
                block_size_q,
                block_size_k,
                topk,
                sm_scale,
                score_type,
                qblock_mappings=qblock_mappings,
                per_query_seq_lens=per_query_seq_lens,
            )
        score = _score_with_metadata(
            indexer,
            q,
            k_cache_bnsd,
            qblock_mappings,
            block_size_q,
            block_size_k,
            sm_scale,
            score_type,
            max_blocks,
        )
        return indexer.flash_prefill_bnsd_topk_from_score(
            score, per_query_seq_lens, topk, block_size_k
        )

    score_topk._sglang_fl_prefill_metadata = True
    MiniMaxSparseAttnBackend._build_prefill_meta = build_metadata
    indexer.flash_prefill_bnsd_with_topk_idx = score_topk
    logger.info("Reusing MiniMax per-forward maximum blocks in prefill indexer")
