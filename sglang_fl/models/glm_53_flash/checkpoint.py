"""Checkpoint naming shared by GLM's BF16 and ModelSlim exports."""


def canonical_weight_name(name: str) -> str:
    """Map the nested ModelSlim/HF modules to the plugin's model layout."""
    return (
        name.replace("model.language_model.", "model.")
        .replace("model.visual.", "visual.")
        .replace(".self_attn.forget_gate.", ".self_attn.")
        .replace(".attn_hc.", ".hc_attn_")
        .replace(".ffn_hc.", ".hc_ffn_")
    )
