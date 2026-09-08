"""Small v0.5.11 compatibility surface used by the GLM-5.3 model.

The vendor reference targets a newer SGLang runtime.  Keep the version bridge
here, in the plugin, so no SGLang source file needs to be edited.
"""

from contextlib import contextmanager
from types import SimpleNamespace


def get_server_args():
    from sglang.srt.server_args import get_global_server_args

    return get_global_server_args()


def get_parallel():
    from sglang.srt.distributed import (
        get_moe_expert_parallel_world_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.layers.dp_attention import (
        get_attention_cp_rank,
        get_attention_cp_size,
        get_attention_tp_group,
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    return SimpleNamespace(
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        attn_tp_rank=get_attention_tp_rank(),
        attn_tp_size=get_attention_tp_size(),
        attn_tp_group=get_attention_tp_group(),
        attn_cp_rank=get_attention_cp_rank(),
        attn_cp_size=get_attention_cp_size(),
        attn_cp_group=get_attention_tp_group(),
        moe_ep_size=get_moe_expert_parallel_world_size(),
    )


class _ForwardContext:
    @contextmanager
    def scoped(self, **_kwargs):
        yield


_FORWARD_CONTEXT = _ForwardContext()


def get_forward():
    return _FORWARD_CONTEXT


def is_deepseek_dsa(config) -> bool:
    archs = getattr(config, "architectures", None) or []
    return (
        "Glm5NextForConditionalGeneration" in archs
        and getattr(config, "index_topk", None) is not None
    )


def is_dsa_enable_prefill_cp() -> bool:
    # GLM KPool does not support prefill CP.  TP16 uses attention TP only.
    return False


def dsa_use_prefill_cp(*_args, **_kwargs) -> bool:
    return False


def mla_use_prefill_cp(*_args, **_kwargs) -> bool:
    return False


def can_dsa_cp_split(*_args, **_kwargs) -> bool:
    return False


def _identity_first(x, *_args, **_kwargs):
    return x


cp_plain_all_gather = _identity_first
cp_plain_reduce_scatter = _identity_first
cp_plain_split = _identity_first
cp_plain_to_scattered = _identity_first
cp_scattered_to_plain = _identity_first
cp_split_and_rebuild_position = _identity_first


class Backend:
    TC_PIECEWISE = "tc_piecewise"


class Phase:
    PREFILL = "prefill"


def check_cuda_graph_backend(*_args, **_kwargs) -> bool:
    return False


def set_attn_hidden_states_local(context, hidden_states) -> None:
    """Rebind attention inputs across the v0.5.11/runtime-context APIs."""

    setter = getattr(context, "set_hidden_states_local", None)
    if setter is not None:
        setter(hidden_states)
    elif context.attn_inputs_ is not None:
        context.attn_inputs_.hidden_states_local = hidden_states


def clear_attn_inputs(context) -> None:
    """Release per-layer attention inputs on both supported context layouts."""

    clear = getattr(context, "clear_attn_inputs", None)
    if clear is not None:
        clear()
    else:
        # v0.5.11 stores the value directly and otherwise only clears it when
        # maybe_input_scattered exits after the whole model forward.
        context.attn_inputs_ = None


# This helper and its API are already present in v0.5.11.  Re-export it from
# the compatibility surface so GLM derives TP-aware dummy heads only after the
# distributed groups have been initialized, and uses the matching weight-pad
# rule during checkpoint loading.
from sglang.srt.layers.attention import vision_utils as vision_utils  # noqa: E402,F401
