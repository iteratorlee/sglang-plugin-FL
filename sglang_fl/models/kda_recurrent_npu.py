"""Ascend varlen recurrent KDA kernel used by GLM-5.3 prefill and decode.

The CANN kernel bundled with the v0.5.11 image has a multi-token KDA gate
pointer bug and implements the default gate/normalization rather than the GLM
formulas.  This plugin-owned variant keeps the same small 910C-friendly tile,
implements the bounded gate and upstream epsilon placement, advances all token
pointers explicitly, and carries state across each complete varlen sequence.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.fla.utils import input_guard


@triton.jit
def _glm_kda_varlen_recurrent_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    lower_bound,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BHV: tl.constexpr,
):
    i_v, i_n, i_nhv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    seq_len = eos - bos

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    for i_bhv in range(BHV):
        i_hv = i_nhv * BHV + i_bhv
        i_h = i_hv // (HV // H)
        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_b = b + bos * HV + i_hv
        p_o = o + (bos * HV + i_hv) * V + o_v
        p_A_log = A_log + i_hv
        p_dt_bias = dt_bias + i_hv * K + o_k

        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias, mask=mask_k).to(tl.float32)
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        state_index = tl.load(h0_indices + i_n)
        p_h0 = (
            h0_source
            + state_index * HV * K * V
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        if state_index > 0:
            b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

        for i in range(seq_len):
            b_q = tl.load(p_q + i * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_k = tl.load(p_k + i * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v + i * HV * V, mask=mask_v, other=0.0).to(tl.float32)
            b_a = tl.load(p_a + i * HV * K, mask=mask_k, other=0.0).to(tl.float32)
            b_beta = tl.load(p_b + i * HV).to(tl.float32)

            decay = tl.exp(b_A_log)
            x = b_a + b_dt_bias
            b_g = lower_bound / (1.0 + tl.exp(-(decay * x)))
            b_beta = 1.0 / (1.0 + tl.exp(-b_beta))

            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q *= scale

            b_h *= tl.exp(b_g[:, None])
            b_v -= tl.sum(b_h * b_k[:, None], axis=0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_q[:, None], axis=0)
            tl.store(
                p_o + i * HV * V,
                b_o.to(p_o.dtype.element_ty),
                mask=mask_v,
            )

        if state_index > 0:
            tl.store(
                p_h0,
                b_h.to(p_h0.dtype.element_ty),
                mask=mask_h,
            )


@input_guard
def glm_kda_varlen_recurrent_npu(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: Optional[float],
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run a packed varlen KDA prefill and update each request state once."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or a.ndim != 4:
        raise ValueError("GLM KDA varlen tensors must have shape [1,T,H,D]")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1 or a.shape[0] != 1:
        raise ValueError("GLM KDA prefill expects one packed token batch")
    if (
        q.shape[:2] != k.shape[:2]
        or q.shape[1] != v.shape[1]
        or q.shape[1] != a.shape[1]
    ):
        raise ValueError("GLM KDA q/k/v/a token dimensions must match")
    if b.shape != v.shape[:-1]:
        raise ValueError("GLM KDA beta must have shape [1,T,HV]")

    num_q_heads, key_dim = k.shape[2:]
    num_value_heads, value_dim = v.shape[2:]
    if a.shape[2:] != (num_value_heads, key_dim):
        raise ValueError("GLM KDA gate must have shape [1,T,HV,K]")
    if num_value_heads % num_q_heads or num_value_heads % 2:
        raise ValueError("GLM KDA requires even HV divisible by H")
    if initial_state_indices.numel() != cu_seqlens.numel() - 1:
        raise ValueError("GLM KDA needs one state index per varlen sequence")
    if lower_bound is None or lower_bound >= 0:
        raise ValueError("GLM KDA requires a negative lower_bound")

    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 64)
    if triton.cdiv(key_dim, block_k) != 1:
        raise NotImplementedError(
            "GLM KDA key dimensions spanning tiles are unsupported"
        )
    if scale is None:
        scale = key_dim**-0.5

    output = torch.zeros_like(v)
    block_value_count = triton.cdiv(value_dim, block_v)
    heads_per_program = 2
    grid = (
        block_value_count,
        cu_seqlens.numel() - 1,
        num_value_heads // heads_per_program,
    )
    _glm_kda_varlen_recurrent_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        o=output,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        lower_bound=lower_bound,
        H=num_q_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        BHV=heads_per_program,
        num_warps=1,
        num_stages=3,
        multibuffer=False,
    )
    return output
