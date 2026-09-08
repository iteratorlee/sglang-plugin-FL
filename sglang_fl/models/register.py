"""Runtime registration and v0.5.11 monkey patches for GLM-5.3."""

import logging
import os
import sys
from contextvars import ContextVar

from transformers import AutoConfig

from .glm5_next_config import Glm5NextConfig, Glm5NextTextConfig

logger = logging.getLogger(__name__)
_REGISTERED = False
_LINEAR_ATTN_REGISTERED = False
_ACTIVE_GLM_CONFIG = ContextVar("sglang_fl_active_glm_config", default=None)
_GLM_POOL_INDEX_HEAD_DIM = ContextVar("sglang_fl_glm_pool_index_head_dim", default=None)


def register_glm5_next() -> None:
    """Register config/model early enough for SGLang config and model discovery."""
    global _REGISTERED, _LINEAR_ATTN_REGISTERED
    if _REGISTERED:
        return

    # ModelRegistry reads this environment variable when its module is first
    # imported.  The external package contains EntryClass in glm5_next.py.
    os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang_fl.models")
    for model_type, cls in (
        ("glm5_next", Glm5NextConfig),
        ("glm5_next_text", Glm5NextTextConfig),
    ):
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:
            # A newer transformers may already provide the same model type.
            pass

    # v0.5.11 deliberately exposes this registry for out-of-tree hybrid
    # models.  Without it ModelRunner only constructs the ordinary Ascend
    # attention backend, so RadixLinearAttention dispatches KDA keyword
    # arguments to AttentionBackend.forward(q, k, v, ...).  Register the
    # text config before ServerArgs/ModelRunner inspect the architecture so
    # the standard AscendHybridLinearAttnBackend owns both MLA and KDA calls.
    if not _LINEAR_ATTN_REGISTERED:
        from sglang.srt.configs.linear_attn_model_registry import (
            LinearAttnModelSpec,
            register_linear_attn_model,
        )

        register_linear_attn_model(
            LinearAttnModelSpec(
                config_class=Glm5NextTextConfig,
                backend_class_name=("sglang_fl.models.ascend_kda.AscendKDAAttnBackend"),
                arch_names=["Glm5NextForConditionalGeneration"],
                uses_mamba_radix_cache=True,
                support_mamba_cache=True,
                support_mamba_cache_extra_buffer=False,
                unwrap_text_config=True,
            )
        )
        _LINEAR_ATTN_REGISTERED = True

    _REGISTERED = True
    logger.info("GLM-5.3 config/model package registered from sglang-plugin-FL")


def patch_deepseek_dsa_compat() -> None:
    """Teach the v0.5.11 MLA implementation to construct its NPU indexer.

    v0.5.11 already contains the NPU DSA execution path but still exposes the
    older NSA config helper names.  Redirect those helpers for GLM without
    editing SGLang.
    """
    try:
        import sglang.srt.models.deepseek_v2 as dsv2
    except Exception:
        return

    def _is_sparse(config):
        archs = getattr(config, "architectures", None) or []
        return (
            "Glm5NextForConditionalGeneration" in archs
            and getattr(config, "index_topk", None) is not None
        )

    def _is_nsa(config):
        original = getattr(patch_deepseek_dsa_compat, "_original_is_nsa", None)
        return _is_sparse(config) or (original(config) if original else False)

    def _is_nsa_for_pool_selection(config):
        """Keep GLM on v0.5.11's hybrid MLA+linear cache path.

        GLM needs the DSA/NSA execution path, but its interleaved KDA layers
        also require ``HybridLinearKVPool``.  The v0.5.11 pool selector tests
        NSA before it tests the hybrid-model branch and would otherwise build
        the generic packed-FP8 ``NSATokenToKVPool``.  That cache is not layout
        compatible with GLM's BF16 KPool-4 index keys.  Only the pool selector
        must see GLM as non-NSA; attention and graph-runner feature detection
        continue to use ``_is_nsa``.
        """

        architectures = getattr(config, "architectures", None) or []
        if "Glm5NextForConditionalGeneration" in architectures:
            return False
        return _is_nsa(config)

    from sglang.srt.configs import model_config as model_config_utils

    if not hasattr(patch_deepseek_dsa_compat, "_original_is_nsa"):
        patch_deepseek_dsa_compat._original_is_nsa = model_config_utils.is_deepseek_nsa
    patch_deepseek_dsa_compat._is_nsa = _is_nsa
    patch_deepseek_dsa_compat._is_nsa_for_pool_selection = (
        _is_nsa_for_pool_selection
    )
    model_config_utils.is_deepseek_nsa = _is_nsa
    model_config_utils.get_nsa_index_head_dim = lambda config: config.index_head_dim
    model_config_utils.get_nsa_index_n_heads = lambda config: config.index_n_heads
    model_config_utils.get_nsa_index_topk = lambda config: config.index_topk
    dsv2.is_deepseek_nsa = _is_nsa
    dsv2.get_nsa_index_head_dim = lambda config: config.index_head_dim
    dsv2.get_nsa_index_n_heads = lambda config: config.index_n_heads
    dsv2.get_nsa_index_topk = lambda config: config.index_topk

    # v0.5.11 constructs the legacy NSA Indexer without passing the model
    # config.  GLM-5.3 needs the later KPool=4 semantics and two extra
    # checkpoint parameters, so route just the duration of this model's
    # attention constructor through the plugin implementation.  ContextVar
    # keeps nested or concurrent model construction isolated.
    if not hasattr(patch_deepseek_dsa_compat, "_original_indexer"):
        patch_deepseek_dsa_compat._original_indexer = dsv2.Indexer
        patch_deepseek_dsa_compat._original_attn_init = (
            dsv2.DeepseekV2AttentionMLA.__init__
        )

        def _indexer_factory(*args, **kwargs):
            config = _ACTIVE_GLM_CONFIG.get()
            if _is_sparse(config) and int(getattr(config, "index_kpool", 1)) > 1:
                from .kpool_indexer import IndexerKPool

                return IndexerKPool(*args, **kwargs, config=config)
            return patch_deepseek_dsa_compat._original_indexer(*args, **kwargs)

        def _attention_init(self, config, *args, **kwargs):
            token = _ACTIVE_GLM_CONFIG.set(config)
            try:
                return patch_deepseek_dsa_compat._original_attn_init(
                    self, config, *args, **kwargs
                )
            finally:
                _ACTIVE_GLM_CONFIG.reset(token)

        dsv2.Indexer = _indexer_factory
        dsv2.DeepseekV2AttentionMLA.__init__ = _attention_init

    # A few v0.5.11 modules bind the helper at import time.  Update only that
    # bound symbol; the wrapper delegates all non-GLM configs to the original.
    for module_name in (
        "sglang.srt.model_executor.model_runner_kv_cache_mixin",
        "sglang.srt.model_executor.pool_configurator",
        "sglang.srt.layers.attention.nsa_backend",
        "sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "is_deepseek_nsa"):
            module.is_deepseek_nsa = (
                _is_nsa_for_pool_selection
                if module_name.endswith("model_runner_kv_cache_mixin")
                else _is_nsa
            )
        if module is not None and hasattr(module, "get_nsa_index_head_dim"):
            module.get_nsa_index_head_dim = lambda config: config.index_head_dim
        if module is not None and hasattr(module, "get_nsa_index_topk"):
            module.get_nsa_index_topk = lambda config: config.index_topk


def patch_glm5_pool_context() -> None:
    """Scope v0.5.11's missing Hybrid NPU MLA index dimension to GLM."""

    # HybridLinearKVPool does not receive the model config, while its v0.5.11
    # NPU MLA constructor also omits index_head_dim. Scope that missing value
    # to the GLM runner's pool initialization instead of retaining process-wide
    # state that could affect a later, unrelated no-RoPE model.
    from sglang.srt.model_executor import model_runner_kv_cache_mixin

    ModelRunnerKVCacheMixin = model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin
    # The module can be imported after ``patch_deepseek_dsa_compat``.  Rebind
    # its from-imported helper here as well so startup order cannot silently
    # select the packed generic NSA pool for GLM.
    pool_predicate = getattr(
        patch_deepseek_dsa_compat, "_is_nsa_for_pool_selection", None
    )
    if pool_predicate is not None:
        model_runner_kv_cache_mixin.is_deepseek_nsa = pool_predicate

    if hasattr(patch_glm5_pool_context, "_original_init_pools"):
        return
    original_init_pools = ModelRunnerKVCacheMixin._init_pools
    patch_glm5_pool_context._original_init_pools = original_init_pools

    def _init_pools(self, *args, **kwargs):
        architectures = (
            getattr(self.model_config.hf_config, "architectures", None) or []
        )
        index_head_dim = (
            int(self.model_config.hf_text_config.index_head_dim)
            if "Glm5NextForConditionalGeneration" in architectures
            else None
        )
        token = _GLM_POOL_INDEX_HEAD_DIM.set(index_head_dim)
        try:
            result = patch_glm5_pool_context._original_init_pools(
                self, *args, **kwargs
            )
            if index_head_dim is not None:
                pool = self.token_to_kv_pool
                full_pool = getattr(pool, "full_kv_pool", None)
                buffers = getattr(full_pool, "index_k_buffer", None)
                if (
                    full_pool is None
                    or not hasattr(full_pool, "get_index_k_buffer")
                    or buffers is None
                    or buffers.shape[0] == 0
                ):
                    raise RuntimeError(
                        "GLM-5.3 requires HybridLinearKVPool with an NPU MLA "
                        "BF16 index cache; "
                        f"outer={type(pool).__module__}.{type(pool).__name__}, "
                        f"full={type(full_pool).__module__}.{type(full_pool).__name__}"
                    )
                logger.info(
                    "GLM-5.3 KV pool ready: outer=%s, full=%s, "
                    "index_dtype=%s, index_shape=%s",
                    type(pool).__name__,
                    type(full_pool).__name__,
                    buffers[0].dtype,
                    tuple(buffers[0].shape),
                )
            return result
        finally:
            _GLM_POOL_INDEX_HEAD_DIM.reset(token)

    ModelRunnerKVCacheMixin._init_pools = _init_pools


def patch_npu_mla_zero_rope_cache() -> None:
    """Give no-RoPE GLM MLA a legal physical Ascend cache shape.

    GLM has logical qk_rope_head_dim=0, while the 0.5.11 Ascend sparse
    attention operator requires a 64-wide RoPE-cache tensor.  Allocate the
    physical placeholder through the existing pool initializer, retain the
    logical dimension, and avoid launching an empty scatter.  Other models use
    the untouched original methods.
    """
    try:
        from sglang.srt.hardware_backend.npu import memory_pool_npu
    except Exception:
        return

    pool_cls = memory_pool_npu.NPUMLATokenToKVPool
    if hasattr(patch_npu_mla_zero_rope_cache, "_original_init"):
        return
    patch_npu_mla_zero_rope_cache._original_init = pool_cls.__init__
    patch_npu_mla_zero_rope_cache._original_set = pool_cls.set_kv_buffer

    def _init(self, *args, **kwargs):
        if "qk_rope_head_dim" in kwargs:
            logical_dim = int(kwargs["qk_rope_head_dim"])
        else:
            # Positional layout: size,page_size,dtype,kv_lora_rank,rope_dim,...
            logical_dim = int(args[4])
        index_head_dim = _GLM_POOL_INDEX_HEAD_DIM.get()
        is_glm_zero_rope = logical_dim == 0 and index_head_dim is not None
        if is_glm_zero_rope:
            if "qk_rope_head_dim" in kwargs:
                kwargs = dict(kwargs)
                kwargs["qk_rope_head_dim"] = 64
            else:
                args = (*args[:4], 64, *args[5:])
        if is_glm_zero_rope and len(args) <= 5 and "index_head_dim" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["index_head_dim"] = index_head_dim
        patch_npu_mla_zero_rope_cache._original_init(self, *args, **kwargs)
        self._sglang_fl_zero_rope = is_glm_zero_rope
        if self._sglang_fl_zero_rope:
            self.qk_rope_head_dim = 0
            self.rope_storage_head_dim = 64

    def _set_kv_buffer(self, layer, loc, cache_k, cache_v):
        if not getattr(self, "_sglang_fl_zero_rope", False):
            return patch_npu_mla_zero_rope_cache._original_set(
                self, layer, loc, cache_k, cache_v
            )
        if cache_v is None:
            cache_k, cache_v = cache_k.split([self.kv_lora_rank, 0], dim=-1)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
        from .graph_ops import scatter_rows_

        layer_buffer = self.k_buffer[layer.layer_id - self.start_layer]
        scatter_rows_(layer_buffer, loc, cache_k)

    pool_cls.__init__ = _init
    pool_cls.set_kv_buffer = _set_kv_buffer


def patch_npu_mla_zero_rope_prepare() -> None:
    """Route only GLM's no-RoPE DSA prepare through the plugin shim."""

    try:
        import sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu as npu_mla
        import sglang.srt.models.deepseek_v2 as dsv2
    except Exception:
        return
    if hasattr(patch_npu_mla_zero_rope_prepare, "_original"):
        return

    from .ascend_dsa import forward_glm5_dsa_prepare_npu

    original = dsv2.forward_dsa_prepare_npu
    patch_npu_mla_zero_rope_prepare._original = original

    def _prepare(m, *args, **kwargs):
        if m.rotary_emb is None and m.qk_rope_head_dim == 0:
            return forward_glm5_dsa_prepare_npu(m, *args, **kwargs)
        return original(m, *args, **kwargs)

    # DeepseekV2AttentionMLA bound the function at module import time.  Patch
    # that bound global as well as the source module, and delegate every
    # non-GLM/RoPE model to the untouched original.
    dsv2.forward_dsa_prepare_npu = _prepare
    npu_mla.forward_dsa_prepare_npu = _prepare


def patch_npu_mla_zero_rope_sparse() -> None:
    """Give GLM's no-RoPE sparse call the physical lanes required by CANN."""

    try:
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )
    except Exception:
        return
    if hasattr(patch_npu_mla_zero_rope_sparse, "_original"):
        return

    from .ascend_dsa import physical_zero_rope

    original = AscendAttnBackend.forward_sparse
    patch_npu_mla_zero_rope_sparse._original = original

    def _forward_sparse(
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
    ):
        pool = forward_batch.token_to_kv_pool
        full_pool = getattr(pool, "full_kv_pool", pool)
        is_glm_zero_rope = (
            self.qk_rope_head_dim == 0
            and getattr(full_pool, "_sglang_fl_zero_rope", False)
        )
        if not is_glm_zero_rope:
            return original(
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
            )
        if q_rope is None or k_rope is None:
            raise ValueError("GLM-5.3 sparse attention requires zero-width RoPE inputs")

        physical_dim = int(getattr(full_pool, "rope_storage_head_dim", 64))
        q_rope, k_rope = physical_zero_rope(q_rope, k_rope, physical_dim)
        # forward_sparse only uses this attribute to reshape the incoming
        # k_rope.  Scope the physical ABI width to this call and restore the
        # logical zero width even if the NPU operator raises.
        self.qk_rope_head_dim = physical_dim
        try:
            return original(
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
            )
        finally:
            self.qk_rope_head_dim = 0

    AscendAttnBackend.forward_sparse = _forward_sparse


def register_glm5_processor() -> None:
    """Reuse the compatible GLM-OCR/GLM4V processor out of tree."""
    try:
        from sglang.srt.managers import multimodal_processor
        from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor

        from .glm5_next import Glm5NextForConditionalGeneration
    except Exception as exc:
        logger.debug("GLM-5.3 processor registration deferred: %s", exc)
        return
    multimodal_processor.PROCESSOR_MAPPING[Glm5NextForConditionalGeneration] = (
        Glm4vImageProcessor
    )


def apply_glm5_patches() -> None:
    register_glm5_next()
    patch_deepseek_dsa_compat()
    patch_glm5_pool_context()
    patch_npu_mla_zero_rope_cache()
    patch_npu_mla_zero_rope_prepare()
    patch_npu_mla_zero_rope_sparse()
    register_glm5_processor()
