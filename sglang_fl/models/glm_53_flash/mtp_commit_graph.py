"""Replay accepted-prefix state copies together without repeated host dispatch."""
import os
import torch
from .mtp_state_npu import commit_state


def state_ops(linear,model):
    cache=linear.req_to_token_pool.get_speculative_mamba2_params_all_layers()
    result=[(cache.temporal,cache.intermediate_ssm,0),
            (cache.conv[0],cache.intermediate_conv_window[0],0)]
    for layer in model.model.layers:
        idx=getattr(layer.self_attn,'indexer',None)
        if idx is not None and hasattr(idx,'_kpool_mtp_tail_k'):
            result.extend(((idx._kpool_tail_k.unsqueeze(0),idx._kpool_mtp_tail_k.unsqueeze(0),1),
                           (idx._kpool_tail_score.unsqueeze(0),idx._kpool_mtp_tail_score.unsqueeze(0),1)))
    return result


def run_ops(ops,indices,req_indices,accepted):
    for destination,source,kind in ops:
        commit_state(destination,source,req_indices if kind else indices,accepted)


class AcceptedStateGraph:
    def __init__(self,ops,indices,req_indices,accepted):
        self.ops=ops
        self.indices=indices.clone();self.req_indices=req_indices.clone();self.accepted=accepted.clone()
        def run():run_ops(self.ops,self.indices,self.req_indices,self.accepted)
        # Copies from immutable verification scratch are idempotent. Warmup
        # and capture therefore commit exactly the same accepted prefix.
        run();torch.npu.synchronize()
        self.graph=torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):run()

    def replay(self,indices,req_indices,accepted):
        self.indices.copy_(indices);self.req_indices.copy_(req_indices);self.accepted.copy_(accepted)
        self.graph.replay()


def commit_accepted(linear,model,indices,req_indices,accepted):
    if os.getenv('SGLANG_FL_GLM53_MTP_COMMIT_GRAPH')!='1' or accepted.numel()==0:
        return run_ops(state_ops(linear,model),indices,req_indices,accepted)
    graphs=getattr(linear,'_glm53_accepted_state_graphs',None)
    if graphs is None:graphs=linear._glm53_accepted_state_graphs={}
    key=(accepted.numel(),indices.dtype,req_indices.dtype)
    if key not in graphs:
        graphs[key]=AcceptedStateGraph(state_ops(linear,model),indices,req_indices,accepted)
    graphs[key].replay(indices,req_indices,accepted)
