"""Ascend graph for the checkpoint's single MTP block after target acceptance."""
from functools import wraps
import logging
import math
import os

_PATCHED = False


def capture_counts(max_requests, tp, draft_tokens):
    """Cover scheduler capacity and make ordinary residual token shards whole."""
    if max_requests < 1 or tp < 1 or draft_tokens < 2:
        raise ValueError("positive request/TP counts and at least two draft tokens required")
    unit = tp // math.gcd(tp, draft_tokens)
    return ((max_requests + tp - 1) // tp * tp,
            (max_requests + unit - 1) // unit * unit)


def capture_batch_sizes(max_requests, tp, draft_tokens, *, dense=False):
    """Capture each legal intermediate MTP size when latency is preferred.

    Draft residuals require request batches divisible by TP; accepted extension
    requires bs*draft_tokens divisible by TP. The existing runner replays the
    smallest captured size >= the actual batch, so no replay change is needed.
    Keep the previous one-size allocation unless explicitly enabled, preserving
    deployments whose HBM budget was validated with the old graph allocation.
    """
    draft_max, extend_max = capture_counts(max_requests, tp, draft_tokens)
    if not dense:
        return [draft_max], [extend_max]
    unit = tp // math.gcd(tp, draft_tokens)
    return list(range(tp, draft_max + 1, tp)), list(range(unit, extend_max + 1, unit))


def patch_draft_extend_graph():
    global _PATCHED
    if _PATCHED: return
    from sglang.srt.speculative.eagle_worker import EAGLEWorker
    original = EAGLEWorker.init_cuda_graphs
    @wraps(original)
    def init(self):
        args = self.server_args
        enabled = (os.getenv('SGLANG_FL_GLM53_MTP_EXTEND_GRAPH') == '1'
            and str(self.device).split(':')[0] == 'npu'
            and args.tp_size == args.ep_size and args.tp_size in (16, 32)
            and args.nnodes == args.tp_size // 16
            and isinstance(args.max_running_requests, int) and args.max_running_requests > 0
            and not args.disable_cuda_graph and not args.enable_dp_attention
            and args.speculative_eagle_topk == 1 and args.speculative_num_steps in (1,2,3,4)
            and type(self.draft_model_runner.model).__name__ == 'Glm5NextForConditionalGenerationNextN')
        if not enabled:
            return original(self)
        from sglang.srt.speculative import eagle_draft_cuda_graph_runner as draft_mod
        previous_sizes = draft_mod.get_batch_sizes_to_capture
        draft_batches, _ = capture_batch_sizes(
            args.max_running_requests, args.tp_size, self.speculative_num_steps + 1,
            dense=os.getenv("SGLANG_FL_GLM53_MTP_DENSE_GRAPHS") == "1")
        def draft_sizes(runner, *args, **kwargs):
            if runner is self.model_runner:
                return draft_batches, []
            return previous_sizes(runner, *args, **kwargs)
        from sglang.srt.hardware_backend.npu.graph_runner import eagle_draft_npu_graph_runner as npu_mod
        from sglang.srt.configs.model_config import is_deepseek_nsa
        npu_mod.is_deepseek_nsa = is_deepseek_nsa
        indexer = self.draft_model_runner.model.model.decoder.self_attn.indexer
        indexer._glm53_draft_decode = True
        draft_mod.get_batch_sizes_to_capture = draft_sizes
        try: original(self)
        finally: draft_mod.get_batch_sizes_to_capture = previous_sizes
        self.cuda_graph_runner_for_draft_extend = GlmDraftExtendGraph(self)
        logging.getLogger(__name__).info(
            "GLM MTP graph capture sizes: draft=%s accepted=%s",
            getattr(self.cuda_graph_runner, "capture_bs", []),
            getattr(self.cuda_graph_runner_for_draft_extend, "capture_bs", []))
        logging.getLogger(__name__).info('GLM MTP accepted-token extend NPU graph captured')
    EAGLEWorker.draft = wrap_draft_transaction(EAGLEWorker.draft)
    EAGLEWorker.init_cuda_graphs = init
    _PATCHED = True


def wrap_draft_transaction(original):
    @wraps(original)
    def draft(self, batch):
        model = self.draft_model_runner.model
        if (type(model).__name__ != 'Glm5NextForConditionalGenerationNextN'
            or self.speculative_num_steps <= 1):
            return original(self, batch)
        indexer = model.model.decoder.self_attn.indexer
        # The draft's speculative forwards must not advance its persistent
        # partial KPool group; accepted extension will write the chosen prefix.
        before_k = indexer._kpool_tail_k.clone()
        before_score = indexer._kpool_tail_score.clone()
        try: return original(self, batch)
        finally:
            indexer._kpool_tail_k.copy_(before_k)
            indexer._kpool_tail_score.copy_(before_score)
    return draft


def GlmDraftExtendGraph(worker):
    import torch
    from sglang.srt.speculative import eagle_draft_extend_cuda_graph_runner as mod
    from sglang.srt.model_executor.cuda_graph_runner import DeepEPCudaGraphRunnerAdapter

    class LowLatencyAdapter(DeepEPCudaGraphRunnerAdapter):
        def capture(self, is_extend_in_batch):
            return super().capture(is_extend_in_batch=False)

    class Runner(mod.EAGLEDraftExtendCudaGraphRunner):
        def _create_graph(self): return torch.npu.NPUGraph()
        def _cache_loc_dtype(self): return torch.int32
        def _capture_graph(self, graph, pool, stream, run_once_fn):
            with torch.npu.graph(graph, pool=pool, stream=stream, auto_dispatch_capture=True):
                return run_once_fn()
        def capture_one_batch_size(self, *args, **kwargs):
            self.deepep_adapter = LowLatencyAdapter()
            self.draft_extend_attn_backend.graph_metadata['block_tables'].zero_()
            return super().capture_one_batch_size(*args, **kwargs)

    # The ordinary residual communicator requires a multiple of TP tokens.
    # Capture a padded request count whose token count is divisible by TP.
    # Cover every request admitted by the scheduler, including capacities above TP.
    original_sizes = mod.get_batch_sizes_to_capture
    _, extend_batches = capture_batch_sizes(
        worker.server_args.max_running_requests, worker.server_args.tp_size,
        worker.speculative_num_steps + 1,
        dense=os.getenv("SGLANG_FL_GLM53_MTP_DENSE_GRAPHS") == "1")
    def sizes(runner, *args, **kwargs):
        if runner is worker.model_runner:
            return extend_batches, []
        return original_sizes(runner, *args, **kwargs)
    indexer = worker.draft_model_runner.model.model.decoder.self_attn.indexer
    indexer._glm53_draft_graph_steps = worker.speculative_num_steps + 1
    mod.get_batch_sizes_to_capture = sizes
    try: return Runner(worker)
    finally: mod.get_batch_sizes_to_capture = original_sizes
