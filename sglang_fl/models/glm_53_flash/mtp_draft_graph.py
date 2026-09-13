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
        original(self)
        args = self.server_args
        if (os.getenv('SGLANG_FL_GLM53_MTP_EXTEND_GRAPH') != '1'
            or str(self.device).split(':')[0] != 'npu'
            or args.tp_size != 16 or args.ep_size != 16 or args.nnodes != 1
            or args.max_running_requests != 1
            or args.disable_cuda_graph or args.enable_dp_attention
            or args.speculative_eagle_topk != 1 or args.speculative_num_steps != 1
            or type(self.draft_model_runner.model).__name__ != 'Glm5NextForConditionalGenerationNextN'):
            return
        self.cuda_graph_runner_for_draft_extend = GlmDraftExtendGraph(self)
        logging.getLogger(__name__).info('GLM MTP accepted-token extend NPU graph captured')
    EAGLEWorker.init_cuda_graphs = init
    _PATCHED = True


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
    # Only this draft runner captures padded BS8 (2 tokens/request); target BS1
    # remains unchanged. At most one real request is admitted by the scheduler.
    original_sizes = mod.get_batch_sizes_to_capture
    def sizes(runner, *args, **kwargs):
        if runner is worker.model_runner:
            return [worker.server_args.tp_size // 2], []
        return original_sizes(runner, *args, **kwargs)
    indexer = worker.draft_model_runner.model.model.decoder.self_attn.indexer
    indexer._glm53_draft_graph_steps = worker.speculative_num_steps + 1
    mod.get_batch_sizes_to_capture = sizes
    try: return Runner(worker)
    finally: mod.get_batch_sizes_to_capture = original_sizes
