"""GLM vision Q/K normalization shares one scale vector across all heads."""
def pad_vision_weight(config, name, weight):
    from .compat import vision_utils
    # VisionAttention(qk_normalization_by_head_size=True) normalizes each
    # head independently. Dummy attention heads do not add scale parameters.
    if name.endswith(('attn.q_norm.weight', 'attn.k_norm.weight')):
        if weight.ndim != 1 or weight.numel() != config.vision_config.head_dim:
            raise ValueError(f'Invalid per-head GLM vision norm: {name} {weight.shape}')
        return weight
    return vision_utils.pad_vit_attn_dummy_heads(config, name, weight)
