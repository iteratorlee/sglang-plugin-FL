"""Ascend graph for the checkpoint's single MTP block after target acceptance."""
from functools import wraps
import logging
import os

_PATCHED = False

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
            and args.tp_size == args.ep_size == 16 and args.nnodes == 1
            and args.max_running_requests in (1, 2)
            and not args.disable_cuda_graph and not args.enable_dp_attention
            and args.speculative_eagle_topk == 1 and args.speculative_num_steps in (1,2,3,4)
            and type(self.draft_model_runner.model).__name__ == 'Glm5NextForConditionalGenerationNextN')
        if not enabled:
            return original(self)
        from sglang.srt.speculative import eagle_draft_cuda_graph_runner as draft_mod
        previous_sizes = draft_mod.get_batch_sizes_to_capture
        def draft_sizes(runner, *args, **kwargs):
            if runner is self.model_runner: return [args_tp], []
            return previous_sizes(runner, *args, **kwargs)
        from sglang.srt.hardware_backend.npu.graph_runner import eagle_draft_npu_graph_runner as npu_mod
        from sglang.srt.configs.model_config import is_deepseek_nsa
        npu_mod.is_deepseek_nsa = is_deepseek_nsa
        args_tp = args.tp_size
        indexer = self.draft_model_runner.model.model.decoder.self_attn.indexer
        indexer._glm53_draft_decode = True
        draft_mod.get_batch_sizes_to_capture = draft_sizes
        try: original(self)
        finally: draft_mod.get_batch_sizes_to_capture = previous_sizes
        self.cuda_graph_runner_for_draft_extend = GlmDraftExtendGraph(self)
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
    # Real request counts remain bounded by the scheduler (one or two).
    original_sizes = mod.get_batch_sizes_to_capture
    def sizes(runner, *args, **kwargs):
        if runner is worker.model_runner:
            import math
            tp = worker.server_args.tp_size
            return [tp // math.gcd(tp, worker.speculative_num_steps + 1)], []
        return original_sizes(runner, *args, **kwargs)
    indexer = worker.draft_model_runner.model.model.decoder.self_attn.indexer
    indexer._glm53_draft_graph_steps = worker.speculative_num_steps + 1
    mod.get_batch_sizes_to_capture = sizes
    try: return Runner(worker)
    finally: mod.get_batch_sizes_to_capture = original_sizes
