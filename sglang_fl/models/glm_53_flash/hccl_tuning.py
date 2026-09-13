"""Scoped GLM communicator settings, applied before lazy HCCL allocation."""
from functools import wraps
import json
import os
from pathlib import Path
from .normal_collective import _glm_checkpoint

_PATCHED = False


def selected_config(args, group_name):
    if group_name not in ('tp', 'moe_ep'):
        return {}
    if not (str(args.device).startswith('npu') and args.tp_size == args.ep_size
            and args.tp_size in (16, 32) and args.nnodes == args.tp_size // 16
            and args.pp_size == 1 and not args.enable_dp_attention
            and not args.enable_two_batch_overlap and args.quantization == 'modelslim'
            and _glm_checkpoint(args.model_path)):
        return {}
    config = {}
    value = os.getenv('SGLANG_FL_GLM53_HCCL_BUFFER_MB')
    if value is not None:
        size = int(value)
        if not 1 <= size <= 4096:
            raise ValueError('GLM HCCL buffer size must be 1..4096 MiB')
        config['hccl_buffer_size'] = size
    return config


def patch_hccl_options():
    global _PATCHED
    if _PATCHED:
        return
    from sglang.srt.distributed import parallel_state
    from sglang.srt.server_args import get_global_server_args
    original = parallel_state.get_torch_distributed_pg_options

    @wraps(original)
    def options(group_name=None):
        result = original(group_name)
        if group_name not in ('tp', 'moe_ep'):
            return result
        if os.getenv('SGLANG_FL_GLM53_HCCL_BUFFER_MB') is None:
            return result
        config = selected_config(get_global_server_args(), group_name)
        if not config:
            return result
        # DeepEP and HCCL must agree on communicator buffer geometry.
        # The launcher must also set this before the world group is created.
        os.environ['HCCL_BUFFSIZE'] = str(config['hccl_buffer_size'])
        import torch_npu
        if result is None:
            result = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        result.hccl_config = dict(result.hccl_config, **config)
        if base := os.getenv('SGLANG_FL_GLM53_AUDIT_DIR'):
            folder = Path(base).parent / 'hccl-options'
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f'{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(dict(rank=parallel_state.get_world_group().rank,
                    group=group_name, hccl_config=result.hccl_config)) + '\n')
        return result

    parallel_state.get_torch_distributed_pg_options = options
    _PATCHED = True
