"""Bootstrap HYV4 config support before SGLang parses server arguments."""

from transformers import AutoConfig, PretrainedConfig

HYV4_SAFE_GRAPH_BATCH_SIZE = 1


class HYV4Config(PretrainedConfig):
    model_type = "hy_v4"

    def __init__(self, **kwargs):
        # Transformers 5.6 validates layer type spellings during base init.
        if kwargs.get("layer_types"):
            kwargs["layer_types"] = [
                "sparse" if item == "deepseek_sparse_attention" else item
                for item in kwargs["layer_types"]
            ]
        quant = kwargs.get("quantization_config")
        if quant and quant.get("quant_method") == "compressed-tensors":
            # The NPU loader supplies packed mappings for gate/up and q_a/kv_a.
            # Do not add the fused dense gate_up module to ``ignore``: this
            # checkpoint contains its INT8 weight_scale tensors and the
            # DeepSeek loader maps both source projections into that module.
            for group in quant.get("config_groups", {}).values():
                targets = group.setdefault("targets", [])
                targets.append("re:^.*\\.shared_experts\\.gate_up_proj$")
                targets.append("re:^.*\\.fused_qkv_a_proj_with_mqa$")
            # Every full indexer checkpoint tensor is BF16. Shared layers
            # deliberately have no indexer tensors because they reuse the
            # preceding full layer's selection, but 0.5.11 still constructs
            # their modules. Keep all of these modules unquantized so target
            # matching succeeds without inventing weights for shared layers.
            indexer_ignore = "re:.*\\.indexer\\..*$"
            ignore = quant.setdefault("ignore", [])
            if indexer_ignore not in ignore:
                ignore.append(indexer_ignore)
        super().__init__(**kwargs)
        # DeepSeek-v2/3 compatibility aliases used by SGLang's MLA/MoE stack.
        self.first_k_dense_replace = 1
        self.moe_layer_freq = 1
        self.n_group = getattr(self, "n_group", 1)
        self.topk_group = getattr(self, "topk_group", 1)
        self.scoring_func = "sigmoid"
        self.seq_aux = False
        self.topk_method = "noaux_tc"
        self.ep_size = 1
        self.num_nextn_predict_layers = getattr(
            self, "num_nextn_predict_layers", 1
        )
        # The NSA implementation uses a frequency/pattern representation.
        indexer_types = getattr(self, "indexer_types", None)
        if indexer_types:
            self.index_topk_pattern = [
                "S" if kind == "shared" else "F" for kind in indexer_types
            ]
        self.index_topk_freq = 4
        self.indexer_rope_interleave = True
        # HYV4's NVIDIA reference calls get_rope(is_neox_style=False).
        # DeepSeek's SGLang adapter derives that flag as !rope_interleave.
        self.rope_interleave = True
        # HYV4 stores the base in the Transformers-5 rope_parameters object,
        # while this SGLang DeepSeek implementation reads a top-level alias.
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        self.rope_theta = rope_parameters.get(
            "rope_theta", getattr(self, "rope_theta", 10_000_000)
        )
        # Keep the checkpoint's native index_topk and position limit.  The
        # Ascend adapter must construct the indexer and execute DSA rather than
        # silently replacing sparse attention with a short-context dense path.
        # Transformers instantiates an empty config while rendering
        # ``to_diff_dict``.  Enforce the checkpoint contract only for a real
        # deserialization, not for that no-argument defaults object.
        if kwargs and getattr(self, "index_topk", None) is None:
            raise ValueError("HYV4 requires checkpoint index_topk for native DSA")
        # The bundled SGLang derives MLA vs. MHA from a hard-coded architecture
        # membership list.  Keep HYV4 first for external model resolution and
        # add a compatible MLA marker so it builds the MLA/NSA KV pool.
        architectures = list(getattr(self, "architectures", None) or [])
        if architectures and architectures[0] == "HYV4ForCausalLM":
            if "DeepseekV3ForCausalLM" not in architectures:
                architectures.append("DeepseekV3ForCausalLM")
            self.architectures = architectures


AutoConfig.register("hy_v4", HYV4Config, exist_ok=True)

_sglang_patches_applied = False


def patch_npu_mla_pool_init(pool_class):
    """Accept 0.5.11's unused NSA keyword once, at bootstrap or worker import."""
    if getattr(pool_class, "_hy_accepts_kv_cache_dim", False):
        return
    original_init = pool_class.__init__

    def init(self, *args, kv_cache_dim=None, **kwargs):
        return original_init(self, *args, **kwargs)

    pool_class.__init__ = init
    pool_class._hy_accepts_kv_cache_dim = True


def apply_sglang_patches():
    """Install 0.5.11 compatibility shims after plugin discovery completes.

    Keeping SGLang imports out of module scope avoids recursively loading the
    ``sglang_fl`` entry point while it is still being initialized.
    """
    global _sglang_patches_applied
    if _sglang_patches_applied:
        return

    from sglang.srt.configs import model_config as model_config
    from sglang.srt.hardware_backend.npu.memory_pool_npu import (
        NPUMLATokenToKVPool,
    )

    old_is_nsa = model_config.is_deepseek_nsa

    def is_hyv4_or_nsa(config):
        archs = (
            config.get("architectures")
            if isinstance(config, dict)
            else getattr(config, "architectures", None)
        )
        index_topk = (
            config.get("index_topk")
            if isinstance(config, dict)
            else getattr(config, "index_topk", None)
        )
        return bool(archs and archs[0] == "HYV4ForCausalLM" and index_topk is not None) or old_is_nsa(config)

    model_config.is_deepseek_nsa = is_hyv4_or_nsa

    old_derive_context_length = model_config.ModelConfig._derive_context_length

    def derive_context_length_with_hyv4_graph_bound(self, context_length):
        architectures = list(getattr(self.hf_config, "architectures", None) or [])
        is_hyv4 = bool(
            architectures and architectures[0] == "HYV4ForCausalLM"
        )
        if is_hyv4:
            # The 2026-09-05 TP32 validation found that identical lanes can be
            # stable at BS2 while a heterogeneous BS2 still corrupts one lane;
            # BS32 also changes request-lane outputs. Only exact BS1 has passed
            # the model-level semantic suite. Refuse every larger capture
            # bucket instead of silently serving incorrect graph results.
            try:
                from sglang.srt.server_args import get_global_server_args

                server_args = get_global_server_args()
            except ValueError:
                # ModelConfig is also constructed directly by unit tests and
                # library users before the scheduler installs global args.
                server_args = None
            if server_args is not None and not server_args.disable_cuda_graph:
                capture_bs = list(server_args.cuda_graph_bs or [])
                if server_args.cuda_graph_max_bs is not None:
                    capture_bs.append(server_args.cuda_graph_max_bs)
                unsafe = [
                    batch_size
                    for batch_size in capture_bs
                    if batch_size > HYV4_SAFE_GRAPH_BATCH_SIZE
                ]
                if unsafe:
                    raise ValueError(
                        "HYV4 Ascend NPUGraph is validated only for exact "
                        f"batch sizes <= {HYV4_SAFE_GRAPH_BATCH_SIZE}; unsafe "
                        f"capture batch sizes requested: {sorted(set(unsafe))}. "
                        "Use --cuda-graph-bs 1 "
                        "--cuda-graph-max-bs 1 --max-running-requests 1."
                    )
                max_running = server_args.max_running_requests
                if (
                    max_running is None
                    or max_running > HYV4_SAFE_GRAPH_BATCH_SIZE
                ):
                    raise ValueError(
                        "HYV4 Ascend NPUGraph requires an explicit "
                        f"max_running_requests <= {HYV4_SAFE_GRAPH_BATCH_SIZE}; "
                        f"got {max_running!r}. Use --max-running-requests 1 "
                        "so every decode batch remains inside the validated "
                        "BS1 graph bucket."
                    )
        return old_derive_context_length(self, context_length)

    model_config.ModelConfig._derive_context_length = (
        derive_context_length_with_hyv4_graph_bound
    )

    patch_npu_mla_pool_init(NPUMLATokenToKVPool)
    _sglang_patches_applied = True
