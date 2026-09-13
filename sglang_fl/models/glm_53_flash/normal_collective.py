"""Opt-in GLM prefill routing through ordinary HCCL collectives."""
from functools import lru_cache, wraps
import os
import torch
import torch.distributed as dist

_PATCHED=False


@lru_cache(maxsize=4)
def _glm_checkpoint(path):
    from pathlib import Path
    import json
    try:
        return 'Glm5NextForConditionalGeneration' in json.loads((Path(path)/'config.json').read_text()).get('architectures',[])
    except (OSError, ValueError, TypeError):
        return False


def collective_dispatch(group, x, ids, weights, num_experts=288):
    world,rank=dist.get_world_size(group),dist.get_rank(group)
    tokens,width=x.shape;topk=ids.shape[1]
    def gather(t):
        out=t.new_empty((world*t.shape[0],*t.shape[1:]))
        dist.all_gather_into_tensor(out,t.contiguous(),group=group)
        return out
    # Row quantization is identical to quantizing the received BF16 rows.
    q,scale=torch.ops.npu.npu_dynamic_quant(x)
    all_q,all_scale,all_ids,all_weights=map(gather,(q,scale,ids,weights))
    flat=all_ids.cpu().flatten()
    assert num_experts % world == 0
    experts=num_experts//world;low=rank*experts
    selected=((flat>=low)&(flat<low+experts)).nonzero().flatten()
    order=torch.argsort(flat[selected],stable=True)
    selected=selected[order]
    counts=torch.bincount(flat[selected]-low,minlength=experts).tolist()
    loc=selected.to(device=x.device,dtype=torch.int64)
    rows=torch.div(loc,topk,rounding_mode='floor')
    if selected.numel():
        recv=all_q.index_select(0,rows)
        recv_scale=all_scale.index_select(0,rows)
    else:
        # Match DeepEP's dummy allocation for a rank with no received tokens.
        recv=q.new_zeros((1,width));recv_scale=scale.new_ones((1,))
    chosen_weights=all_weights.flatten().index_select(0,loc)
    state=(rows,chosen_weights,tokens,world,width,int(selected.numel()))
    return recv,recv_scale,counts,state


def collective_combine(group, y, state):
    rows,weights,tokens,world,width,received=state
    # Sum each rank's expert contributions in FP32, then reduce them across EP.
    # This keeps the workspace bounded by tokens*hidden, rather than topk times
    # that size. The final activation is BF16; expert products remain W8A8.
    out=torch.zeros((world*tokens,width),device=y.device,dtype=torch.float32)
    if received:
        out.index_add_(0,rows,y[:received].float()*weights[:,None])
    dist.all_reduce(out,group=group)
    rank=dist.get_rank(group)
    return out.narrow(0,rank*tokens,tokens).to(y.dtype)


def enabled(args):
    return (os.getenv('SGLANG_FL_GLM53_NORMAL_HCCL')=='1'
            and str(args.device).startswith('npu') and args.tp_size==args.ep_size and args.tp_size in (16,32)
            and args.nnodes==args.tp_size//16 and args.pp_size==1 and not args.enable_dp_attention
            and not args.enable_two_batch_overlap and not args.enable_eplb
            and args.ep_num_redundant_experts==0 and args.quantization=='modelslim'
            and _glm_checkpoint(args.model_path))


def patch_normal_collectives():
    global _PATCHED
    if _PATCHED:return
    from sglang.srt.layers.moe.token_dispatcher.deepep import _DeepEPDispatcherImplNormal
    original_dispatch=_DeepEPDispatcherImplNormal._dispatch_core
    original_combine=_DeepEPDispatcherImplNormal._combine_core
    @wraps(original_dispatch)
    def dispatch(self,x,ids,weights,event):
        from sglang.srt.server_args import get_global_server_args
        if not enabled(get_global_server_args()):return original_dispatch(self,x,ids,weights,event)
        if os.getenv("DEEP_NORMAL_MODE_USE_INT8_QUANT") != "1":
            raise ValueError("GLM HCCL routing requires DEEP_NORMAL_MODE_USE_INT8_QUANT=1")
        from deep_ep.utils import EventOverlap
        recv,scale,counts,self._glm53_collective_state=collective_dispatch(self.group,x,ids,weights,self.num_experts)
        return (recv,scale),ids,weights,counts,EventOverlap()
    @wraps(original_combine)
    def combine(self,x,event):
        state=getattr(self,'_glm53_collective_state',None)
        if state is None:return original_combine(self,x,event)
        from deep_ep.utils import EventOverlap
        out=collective_combine(self.group,x,state)
        self._glm53_collective_state=None
        return out,EventOverlap()
    _DeepEPDispatcherImplNormal._dispatch_core=dispatch
    _DeepEPDispatcherImplNormal._combine_core=combine
    _PATCHED=True
