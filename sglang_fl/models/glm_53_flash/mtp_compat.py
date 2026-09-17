"""GLM-only integration with v0.5.11's EAGLE scheduler and model loader."""

from functools import wraps
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)
TARGET = 'Glm5NextForConditionalGeneration'
DRAFT = TARGET + 'NextN'
_PATCHED = False


def patch_mtp():
    global _PATCHED
    if _PATCHED:
        return
    from sglang.srt.configs.model_config import ModelConfig, AttentionArch
    original_draft = ModelConfig._config_draft_model
    original_shapes = ModelConfig._derive_model_shapes

    @wraps(original_draft)
    def configure(self):
        original_draft(self)
        if self.is_draft_model and TARGET in self.hf_config.architectures:
            text = self.hf_text_config
            text.glm53_nextn_layer_id = text.num_hidden_layers
            text.num_hidden_layers = 1
            text.linear_attn_config = dict(text.linear_attn_config,
                kda_layers=[], full_attn_layers=[0])
            self.hf_config.architectures = [DRAFT]
            text.architectures = [DRAFT]

    @wraps(original_shapes)
    def shapes(self):
        original_shapes(self)
        if any(a in (TARGET, DRAFT) for a in self.hf_config.architectures):
            cfg = self.hf_text_config
            # GLM's layer communicator contracts mHC before the final norm.
            self.spec_hidden_size = cfg.hidden_size
            self.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = cfg.kv_lora_rank
            self.qk_nope_head_dim = cfg.qk_nope_head_dim
            self.qk_rope_head_dim = cfg.qk_rope_head_dim
            self.v_head_dim = cfg.v_head_dim
            self.index_head_dim = cfg.index_head_dim
            self.scaling = self.head_dim ** -0.5

    ModelConfig._config_draft_model = configure
    ModelConfig._derive_model_shapes = shapes

    # Read only the actual MTP tensors; do not materialize all target shards.
    from sglang.srt.model_loader.loader import DefaultModelLoader
    original_weights = DefaultModelLoader._get_all_weights

    @wraps(original_weights)
    def weights(self, model_config, model):
        if DRAFT not in model_config.hf_config.architectures:
            yield from original_weights(self, model_config, model)
            return
        from safetensors import safe_open
        folder = Path(model_config.model_path)
        index = json.loads((folder/'quant_model_weights.safetensors.index.json').read_text())['weight_map']
        layer = model.config.num_hidden_layers
        prefixes = (f'model.layers.{layer}.', f'model.language_model.layers.{layer}.')
        by_file = {}
        for name, file in index.items():
            if name.startswith(prefixes) or name == 'rot.weight':
                by_file.setdefault(file, []).append(name)
        if not by_file:
            raise ValueError('No GLM MTP tensors found in checkpoint index')
        logger.info('GLM MTP loader selects %d tensors from %d shards',
                    sum(map(len, by_file.values())), len(by_file))
        for file, names in sorted(by_file.items()):
            with safe_open(str(folder/file), framework='pt', device='cpu') as handle:
                for name in sorted(names):
                    yield name, handle.get_tensor(name)

    DefaultModelLoader._get_all_weights = weights

    # Generic Kimi state layout is [window, dimension]. The image's NPU pool
    # swaps these axes; GLM stores [dimension, 3] and snapshots separately.
    from sglang.srt.hardware_backend.npu import memory_pool_npu
    from .register import _GLM_POOL_INDEX_HEAD_DIM
    original_conv = memory_pool_npu._init_npu_conv_state

    @wraps(original_conv)
    def conv_state(conv_state_in, conv_state_shape, speculative_num_draft_tokens=None):
        if _GLM_POOL_INDEX_HEAD_DIM.get() is not None:
            speculative_num_draft_tokens = None
        return original_conv(conv_state_in, conv_state_shape, speculative_num_draft_tokens)

    memory_pool_npu._init_npu_conv_state = conv_state

    from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend
    original_commit = HybridLinearAttnBackend.update_mamba_state_after_mtp_verify

    @wraps(original_commit)
    def commit(self, accepted_steps, mamba_track_indices, mamba_steps_to_track, model):
        from .ascend_kda import AscendKDAAttnBackend
        if not isinstance(self.linear_attn_backend, AscendKDAAttnBackend):
            return original_commit(self, accepted_steps, mamba_track_indices, mamba_steps_to_track, model)
        if mamba_track_indices is not None:
            raise ValueError('GLM MTP requires disabled radix cache')
        from .mtp_commit_graph import commit_accepted
        linear = self.linear_attn_backend
        indices = linear.forward_metadata.mamba_cache_indices[:accepted_steps.numel()]
        req_indices = linear.verify_req_pool_indices[:accepted_steps.numel()]
        return commit_accepted(linear, model, indices, req_indices, accepted_steps)

    HybridLinearAttnBackend.update_mamba_state_after_mtp_verify = commit
    _PATCHED = True


_EAGLE_PATCHED = False


def wrap_glm_verify(original):
    """v0.5.11's EAGLE commit condition omits the registered KDA family."""
    @wraps(original)
    def verify(self, batch, spec_info):
        result = original(self, batch, spec_info)
        runner = self.target_worker.model_runner
        from .ascend_kda import AscendKDAAttnBackend
        linear = getattr(runner.attn_backend, 'linear_attn_backend', None)
        if isinstance(linear, AscendKDAAttnBackend) and not any(
            getattr(runner, name, None) is not None for name in
            ('hybrid_gdn_config', 'mamba2_config', 'hybrid_lightning_config')
        ):
            import torch
            accepted = torch.tensor(result[1].num_accepted_drafts_per_req_cpu,
                                    device=runner.device, dtype=torch.int64)
            if accepted.numel():
                runner.attn_backend.update_mamba_state_after_mtp_verify(
                    accepted_steps=accepted, mamba_track_indices=None,
                    mamba_steps_to_track=None, model=runner.model)
        return result
    return verify


def patch_eagle_verify():
    global _EAGLE_PATCHED
    if _EAGLE_PATCHED:
        return
    from sglang.srt.speculative.eagle_worker import EAGLEWorker
    from .mtp_sampling import patch_mtp_sampling
    patch_mtp_sampling()
    EAGLEWorker.verify = wrap_glm_verify(EAGLEWorker.verify)
    _EAGLE_PATCHED = True
