# Copyright 2026 FlagOS Contributors
"""GDN decode bridge; optimized kernels are owned by FlagGems."""

from sglang_fl.dispatch import call_op


def fused_sigmoid_gating_delta_rule_update_bridge(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    is_kda=False,
):
    return call_op(
        "fused_sigmoid_gating_delta_rule_update",
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        is_kda=is_kda,
    )
