"""Vendor monkey-patches on sglang internals for Ascend / NPU — entrypoint.

These replace direct edits to sglang source that were previously required on
Huawei NPU:
  - scheduler_pp: PP send/recv ordering + stream syncs (HCCL deadlock fix)
  - attention_registry: defer CUDA-only linear-attn backend imports on NPU
  - qwen_vl_processor: transformers-version-compatible Qwen-VL preprocess
  - flagtree_driver: cache lazy driver utilities in affected FlagTree versions
"""

import logging

from .patches.attention_registry import patch_attn_backend_wrapper
from .patches.flagtree_driver import patch_flagtree_driver_utils
from .patches.qwen_vl_processor import patch_qwen_vl_processor
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

    patch_flagtree_driver_utils()
    patch_pp_send_recv_order()
    patch_pp_launch_batch_sync()
    patch_attn_backend_wrapper()
    patch_qwen_vl_processor()


apply_ascend_patches()
