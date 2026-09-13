"""Allow small GLM target TP graphs with ceil-divided MoE slices."""

from contextvars import ContextVar
from functools import wraps
import torch

_IN_SMALL_GRAPH = ContextVar("glm53_small_graph", default=False)
_PATCHED = False


def _enabled(runner):
    args = runner.server_args
    arch = getattr(runner.model_config.hf_config, "architectures", None) or []
    return (
        "Glm5NextForConditionalGeneration" in arch
        and str(runner.device).startswith("npu")
        and not args.enable_dp_attention
        and (not args.speculative_algorithm or (args.speculative_algorithm == "EAGLE"
             and getattr(args, "speculative_eagle_topk", 0) == 1))
        and not args.enable_two_batch_overlap
        and any(0 < bs < args.tp_size for bs in args.cuda_graph_bs)
    )


def patch_small_graphs():
    global _PATCHED
    if _PATCHED:
        return
    from sglang.srt.model_executor import cuda_graph_runner as mod
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    original_sizes = mod.get_batch_sizes_to_capture
    original_count = mod.compute_local_num_token_non_padded

    @wraps(original_sizes)
    def batch_sizes(runner, num_tokens_per_bs=1):
        if not _enabled(runner):
            return original_sizes(runner, num_tokens_per_bs)
        args = runner.server_args
        limit = runner.req_to_token_pool.size
        sizes = sorted({bs for bs in args.cuda_graph_bs if 0 < bs <= limit})
        if not sizes:
            return original_sizes(runner, num_tokens_per_bs)
        return sizes, ([bs for bs in sizes if bs <= args.torch_compile_max_bs]
                       if args.enable_torch_compile else [])

    @wraps(original_count)
    def local_count(global_num_token_non_padded, num_tokens_per_dp):
        if not _IN_SMALL_GRAPH.get():
            return original_count(global_num_token_non_padded, num_tokens_per_dp)
        size = get_attention_tp_size()
        chunk = (num_tokens_per_dp + size - 1) // size
        return torch.clamp(
            global_num_token_non_padded - chunk * get_attention_tp_rank(), 0, chunk
        )

    def scoped(original):
        @wraps(original)
        def wrapped(self, *args, **kwargs):
            token = _IN_SMALL_GRAPH.set(_enabled(self.model_runner))
            try:
                return original(self, *args, **kwargs)
            finally:
                _IN_SMALL_GRAPH.reset(token)

        return wrapped

    mod.get_batch_sizes_to_capture = batch_sizes
    mod.compute_local_num_token_non_padded = local_count
    cls = mod.CudaGraphRunner
    cls.capture_one_batch_size = scoped(cls.capture_one_batch_size)
    cls.replay_prepare = scoped(cls.replay_prepare)
    _PATCHED = True
