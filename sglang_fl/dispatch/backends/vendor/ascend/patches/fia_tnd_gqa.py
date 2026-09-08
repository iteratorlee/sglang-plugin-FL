"""Route a safe fresh FIA prefill shape through the TND GQA interface.

The stock Ascend backend invokes FIA once per request with BSND because older
CANN releases did not accept TND when query and KV head counts differed.  The
installed operator accepts TND+GQA for a fully-fresh EXTEND batch.  This patch
keeps the stock path for every other mode or shape.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)
_MARKER = "_sglang_fl_fia_tnd_gqa"


def _cpu_ints(value):
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                return None
            value = value.tolist()
        return [int(item) for item in value]
    except (TypeError, ValueError, RuntimeError):
        return None


def _eligible(
    self,
    q,
    k,
    v,
    layer,
    forward_batch,
    save_kv_cache,
    q_rope,
    k_rope,
    topk_indices,
    sinks,
    slopes,
):
    """Only select an unpadded, all-new EXTEND batch."""
    if os.getenv("SGLANG_FL_FIA_TND_GQA", "0") != "1":
        return None
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
    except ImportError:
        return None
    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return None
    if not getattr(self, "use_fia", False) or getattr(self, "use_mla", False):
        return None
    cp_size = getattr(self, "attn_cp_size", 1)
    if getattr(self, "is_dllm_model", False) or (cp_size is not None and cp_size > 1):
        return None
    if getattr(self, "use_alibi", False):
        return None
    attn_type = getattr(layer, "attn_type", None)
    if getattr(attn_type, "name", None) == "ENCODER_ONLY":
        return None
    if getattr(layer, "sliding_window_size", -1) not in (-1, None):
        return None
    if getattr(layer, "logit_cap", 0) != 0:
        return None
    if (
        q_rope is not None
        or k_rope is not None
        or topk_indices is not None
        or sinks is not None
        or slopes is not None
    ):
        return None
    if not save_kv_cache or q is None or k is None or v is None:
        return None
    if getattr(forward_batch, "encoder_lens", None) is not None:
        return None
    if getattr(forward_batch, "attn_cp_metadata", None) is not None:
        return None
    if getattr(layer, "is_cross_attention", False):
        return None

    extend = _cpu_ints(getattr(forward_batch, "extend_seq_lens_cpu", None))
    seq_lens = _cpu_ints(getattr(forward_batch, "seq_lens_cpu", None))
    batch_size = getattr(forward_batch, "batch_size", None)
    if (
        not extend
        or seq_lens is None
        or batch_size != 1
        or len(extend) != 1
        or len(seq_lens) != 1
    ):
        return None
    # A continuation has seq_len > extend_len; a mixed/padded batch fails this
    # equality and stays on the stock per-request BSND path.
    if extend[0] != 16384 or seq_lens != extend:
        return None
    prefix_lens = _cpu_ints(getattr(forward_batch, "extend_prefix_lens_cpu", None))
    if prefix_lens is not None and any(prefix_lens):
        return None

    try:
        q_heads = int(layer.tp_q_head_num)
        kv_heads = int(layer.tp_k_head_num)
        v_heads = int(getattr(layer, "tp_v_head_num", kv_heads))
        qk_dim = int(layer.qk_head_dim)
        v_dim = int(layer.v_head_dim)
    except (AttributeError, TypeError, RuntimeError, ValueError):
        return None
    # This is an operator-verified shape guard, not a model-name check.  It
    # prevents the 35B and other head layouts from silently taking this path.
    if (q_heads, kv_heads, v_heads, qk_dim, v_dim) != (12, 2, 2, 256, 256):
        return None
    if not (q.device == k.device == v.device):
        return None
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return None
    if not (q.is_contiguous() and k.is_contiguous()):
        return None
    try:
        q_tnd = q.view(-1, q_heads, qk_dim)
        k_tnd = k.view(-1, kv_heads, qk_dim)
        v_tnd = v.view(-1, v_heads, v_dim)
    except (RuntimeError, ValueError):
        return None
    tokens = sum(extend)
    if q_tnd.shape[0] != tokens or k_tnd.shape[0] != tokens or v_tnd.shape[0] != tokens:
        return None
    # V is a zero-copy slice of packed [Q, gate, K, V]. FIA accepts the
    # verified strided layout; requiring a contiguous V excludes the model.
    if v_tnd.stride() not in ((512, 256, 1), (7168, 256, 1)):
        return None
    if q_tnd.device.type != "npu" or q_tnd.dtype != torch.bfloat16:
        return None
    mask = getattr(self, "fia_mask", None)
    if (
        mask is None
        or mask.ndim != 2
        or tuple(mask.shape) != (2048, 2048)
        or mask.dtype != torch.bool
        or mask.device != q_tnd.device
        or not mask.is_contiguous()
    ):
        return None
    cumulative = []
    total = 0
    for length in extend:
        total += length
        cumulative.append(total)
    return q_tnd, k_tnd, v_tnd, q_heads, kv_heads, cumulative, mask


def patch_fia_tnd_gqa() -> bool:
    """Install the opt-in TND path while leaving graph/decode untouched."""
    if os.getenv("SGLANG_FL_FIA_TND_GQA", "0") != "1":
        return False
    try:
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )
    except (ImportError, RuntimeError):
        return False
    original = AscendAttnBackend.forward_extend
    if getattr(original, _MARKER, False):
        return False
    expected_parameters = (
        "self",
        "q",
        "k",
        "v",
        "layer",
        "forward_batch",
        "save_kv_cache",
        "q_rope",
        "k_rope",
        "topk_indices",
        "sinks",
        "slopes",
    )
    try:
        compatible = (
            tuple(inspect.signature(original).parameters) == expected_parameters
        )
    except (TypeError, ValueError):
        compatible = False
    if not compatible:
        logger.warning("Skipping FIA TND patch: incompatible forward_extend signature")
        return False

    @functools.wraps(original)
    def wrapped(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        topk_indices=None,
        sinks=None,
        slopes=None,
    ):
        selected = _eligible(
            self,
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache,
            q_rope,
            k_rope,
            topk_indices,
            sinks,
            slopes,
        )
        if selected is None:
            return original(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
                topk_indices=topk_indices,
                sinks=sinks,
                slopes=slopes,
            )
        q_tnd, k_tnd, v_tnd, q_heads, kv_heads, cumulative, mask = selected
        # Keep the stock cache visibility/order: KV is committed before the
        # attention result is returned to the layer.
        cache_loc = forward_batch.out_cache_loc
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
        # Never hide an operator/OOM/device failure by retrying after KV writes.
        attn_output = torch.ops.npu.npu_fused_infer_attention_score(
            q_tnd,
            k_tnd,
            v_tnd,
            num_heads=q_heads,
            num_key_value_heads=kv_heads,
            input_layout="TND",
            atten_mask=mask,
            sparse_mode=3,
            scale=layer.scaling,
            next_tokens=0,
            actual_seq_lengths=cumulative,
            actual_seq_lengths_kv=cumulative,
        )[0]
        expected = (q_tnd.shape[0], q_heads, v_tnd.shape[-1])
        if tuple(attn_output.shape) != expected or attn_output.dtype != q.dtype:
            raise RuntimeError(
                f"unexpected TND FIA output {tuple(attn_output.shape)} {attn_output.dtype}"
            )
        if not getattr(self, "_sglang_fl_fia_tnd_hit_logged", False):
            logger.info(
                "FIA_TND_FAST_HIT: tokens=%s q_heads=%s kv_heads=%s",
                cumulative[-1],
                q_heads,
                kv_heads,
            )
            self._sglang_fl_fia_tnd_hit_logged = True
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()
        return attn_output.view(-1, q_heads * v_tnd.shape[-1])

    setattr(wrapped, _MARKER, True)
    AscendAttnBackend.forward_extend = wrapped
    logger.info("Installed opt-in FIA TND+GQA fresh-prefill path")
    return True
