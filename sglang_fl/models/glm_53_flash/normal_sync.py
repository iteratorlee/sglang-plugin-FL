"""Scoped synchronization for GLM normal DeepEP dispatch on the installed NPU runtime."""
from functools import lru_cache, wraps
import json
import os
from pathlib import Path

_PATCHED=False

@lru_cache(maxsize=4)
def _checkpoint(path):
    try:
        return 'Glm5NextForConditionalGeneration' in json.loads((Path(path)/'config.json').read_text()).get('architectures',[])
    except (OSError, ValueError, TypeError):
        return False

def enabled(args):
    return (os.getenv('SGLANG_FL_GLM53_DEEPEP_SYNC')=='1'
            and str(args.device).startswith('npu')
            and args.tp_size==args.ep_size==16 and args.nnodes==args.pp_size==1
            and not args.enable_dp_attention and args.quantization=='modelslim'
            and _checkpoint(args.model_path))

def wrap_dispatch(original, should_sync, synchronize):
    @wraps(original)
    def dispatch(self,*args,**kwargs):
        if not should_sync():return original(self,*args,**kwargs)
        synchronize()
        result=original(self,*args,**kwargs)
        synchronize()
        return result
    return dispatch

def patch_normal_dispatch_sync():
    global _PATCHED
    if _PATCHED:return
    import torch
    from deep_ep import Buffer
    from sglang.srt.server_args import get_global_server_args
    Buffer.dispatch=wrap_dispatch(Buffer.dispatch,
        lambda:enabled(get_global_server_args()),lambda:torch.npu.synchronize())
    _PATCHED=True
