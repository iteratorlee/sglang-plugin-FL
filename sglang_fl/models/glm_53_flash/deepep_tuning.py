"""Bound GLM's low-latency receive padding for small Ascend deployments."""

import logging
import os

logger = logging.getLogger(__name__)
_CAPACITY_ENV = 'SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK'


def small_batch_capacity(args, attention_tp_size):
    """Conservative capacity for the tested 16-card, at-most-four-slot shapes.

    Eager decode may round the attention batch to TP; speculative verification
    can put draft_token_num tokens in each request. Prefill keeps normal DeepEP.
    """
    requests = args.max_running_requests
    if (args.tp_size != 16 or args.ep_size != 16
        or args.moe_a2a_backend != 'deepep'
        or str(args.deepep_mode).lower() != 'auto'
        or requests is None or not 1 <= requests <= 4
        or args.enable_two_batch_overlap):
        return 128
    dp = args.dp_size if args.enable_dp_attention else 1
    local_requests = (requests + dp - 1) // dp
    graph_batches = args.cuda_graph_bs or [local_requests]
    # SGLang caps captured batches to its rounded request-pool capacity.
    rounded_pool = ((local_requests + attention_tp_size - 1)
                    // attention_tp_size * attention_tp_size)
    largest = min(max(graph_batches), rounded_pool)
    largest = max(largest, local_requests)
    padded_batch = ((largest + attention_tp_size - 1)
                    // attention_tp_size * attention_tp_size)
    draft_tokens = args.speculative_num_draft_tokens or 1
    needed = padded_batch * draft_tokens // attention_tp_size
    return next((cap for cap in (1, 2, 4, 16, 128) if cap >= needed), 128)


def configure_small_batch_capacity(args, attention_tp_size):
    if _CAPACITY_ENV in os.environ:
        return
    capacity = small_batch_capacity(args, attention_tp_size)
    if capacity < 128:
        os.environ[_CAPACITY_ENV] = str(capacity)
        logger.info('GLM DeepEP low-latency capacity: %d tokens per rank', capacity)
