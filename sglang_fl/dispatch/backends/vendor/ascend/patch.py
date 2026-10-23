"""Vendor monkey-patches on sglang internals for Ascend / NPU — entrypoint.

These replace direct edits to sglang source that were previously required on
Huawei NPU:
  - scheduler_pp: PP send/recv ordering + stream syncs (HCCL deadlock fix)
  - attention_registry: defer CUDA-only linear-attn backend imports on NPU
  - qwen_vl_processor: transformers-version-compatible Qwen-VL preprocess
"""

import logging

from .patches.attention_registry import patch_attn_backend_wrapper
from .patches.graph_stages import patch_ascend_graph_stages
from .patches.graph_updates import patch_ascend_graph_updates
from .patches.hccl_tp import patch_ascend_tp_hccl_options
from .patches.minimax_m3_image import (
    patch_minimax_m3_image_grid_cache_key,
    patch_minimax_m3_image_preprocess_device,
    patch_minimax_m3_image_token_boundaries,
    patch_minimax_m3_vision_rope_compile,
)
from .patches.minimax_m3_indexer import patch_ascend_minimax_vector_indexer
from .patches.minimax_m3_score_buffers import patch_ascend_minimax_score_buffers
from .patches.minimax_m3_prefill_causal import patch_ascend_minimax_prefill_causal
from .patches.minimax_m3_prefill_future import patch_ascend_minimax_prefill_future
from .patches.minimax_m3_kv_pool import patch_minimax_m3_kv_pool
from .patches.minimax_m3_decode_ifa import patch_minimax_m3_decode_ifa
from .patches.minimax_m3_kv_store import patch_minimax_m3_kv_store
from .patches.minimax_m3_graph_audit import patch_minimax_m3_graph_audit
from .patches.minimax_m3_moe import (
    patch_minimax_m3_moe_activation,
    patch_minimax_m3_w8a8_moe_offsets,
    patch_minimax_m3_w8a8_quant_targets,
)
from .patches.profiler import patch_ascend_profiler_activity
from .patches.qwen_vl_processor import patch_qwen_vl_processor
from .patches.sampling import patch_ascend_torch_sampling
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
    patch_ascend_profiler_activity()
    patch_ascend_tp_hccl_options()
    patch_qwen_vl_processor()
    patch_ascend_torch_sampling()
    patch_ascend_graph_updates()
    patch_ascend_graph_stages()
    patch_ascend_minimax_vector_indexer()
    patch_ascend_minimax_score_buffers()
    patch_ascend_minimax_prefill_causal()
    patch_ascend_minimax_prefill_future()
    patch_minimax_m3_moe_activation()
    patch_minimax_m3_w8a8_moe_offsets()
    patch_minimax_m3_w8a8_quant_targets()
    patch_minimax_m3_image_preprocess_device()
    patch_minimax_m3_image_token_boundaries()
    patch_minimax_m3_image_grid_cache_key()
    patch_minimax_m3_vision_rope_compile()
    patch_minimax_m3_kv_pool()
    patch_minimax_m3_decode_ifa()
    patch_minimax_m3_kv_store()
    patch_minimax_m3_graph_audit()


apply_ascend_patches()
