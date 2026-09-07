"""Vendor monkey-patches on sglang internals for Ascend / NPU — entrypoint.

These replace direct edits to sglang source that were previously required on
Huawei NPU:
  - scheduler_pp: PP send/recv ordering + stream syncs (HCCL deadlock fix)
  - attention_registry: defer CUDA-only linear-attn backend imports on NPU
  - qwen_vl_processor: transformers-version-compatible Qwen-VL preprocess
  - qwen36_causal_conv: batch equal-length multi-sequence prefill calls
  - qwen36_causal_conv_single: skip singleton wrapper list/cat on prefill
  - qwen36_causal_conv_decode_fused: fuse BF16 decode conv and state update
  - vocab_parallel_embedding: fuse sharded lookup and zero masking
"""

import logging

from .patches.attention_registry import patch_attn_backend_wrapper
from .patches.qwen36_causal_conv import patch_qwen36_causal_conv_prefill
from .patches.qwen36_causal_conv_single import patch_qwen36_causal_conv_single
from .patches.qwen36_causal_conv_decode_fused import patch_qwen36_causal_conv_decode_fused
from .patches.qwen_vl_processor import patch_qwen_vl_processor
from .patches.vocab_parallel_embedding import patch_vocab_parallel_embedding
from .patches.scheduler_pp import (
    patch_pp_launch_batch_sync,
    patch_pp_send_recv_order,
)

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_ascend_patches() -> None:
    """Apply all Ascend/NPU-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_pp_send_recv_order()
    patch_pp_launch_batch_sync()
    patch_attn_backend_wrapper()
    patch_qwen_vl_processor()
    patch_qwen36_causal_conv_prefill()
    patch_qwen36_causal_conv_single()
    patch_qwen36_causal_conv_decode_fused()
    patch_vocab_parallel_embedding()


apply_ascend_patches()
