"""Ascend graph-safe fused mHC; FP32 projection and mixing arithmetic."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _hc_coefficients(
    X,
    MIX,
    SCALE,
    BASE,
    PRE,
    POST,
    COMB,
    TOTAL: tl.constexpr,
    BT: tl.constexpr,
    M: tl.constexpr,
    RMS_EPS: tl.constexpr,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
    POST_MULT: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.arange(0, BT)
    x = tl.load(X + t * TOTAL + k, k < TOTAL, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / TOTAL + RMS_EPS)
    i = tl.arange(0, M)
    stride: tl.constexpr = 2 * M + M * M
    a = tl.load(MIX + t * stride + i) * inv
    b = tl.load(MIX + t * stride + M + i) * inv
    a = a * tl.load(SCALE) + tl.load(BASE + i)
    b = b * tl.load(SCALE + 1) + tl.load(BASE + M + i)
    pre = 1.0 / (1.0 + tl.exp(-a)) + EPS
    post = POST_MULT / (1.0 + tl.exp(-b))
    ij = i[:, None] * M + i[None, :]
    c = tl.load(MIX + t * stride + 2 * M + ij) * inv
    c = c * tl.load(SCALE + 2) + tl.load(BASE + 2 * M + ij)
    c = tl.exp(c - tl.max(c, 1)[:, None])
    c = c / tl.sum(c, 1)[:, None] + EPS
    c = c / (tl.sum(c, 0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        c = c / (tl.sum(c, 1)[:, None] + EPS)
        c = c / (tl.sum(c, 0)[None, :] + EPS)
    tl.store(PRE + t * M + i, pre)
    tl.store(POST + t * M + i, post)
    tl.store(COMB + t * M * M + ij, c)


@triton.jit
def _hc_pre_apply(X, PRE, OUT, H: tl.constexpr, M: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    i = tl.arange(0, M)
    x = tl.load(X + t * M * H + i[:, None] * H + h[None, :], h[None, :] < H, 0).to(
        tl.float32
    )
    pre = tl.load(PRE + t * M + i)
    out = tl.sum(x * pre[:, None], 0)
    tl.store(OUT + t * H + h, out, h < H)


@triton.jit
def _hc_post_apply(
    X, RES, POST, COMB, OUT, H: tl.constexpr, M: tl.constexpr, B: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    i = tl.arange(0, M)
    x = tl.load(X + t * H + h, h < H, 0).to(tl.float32)
    residual = tl.load(
        RES + t * M * H + i[:, None] * H + h[None, :], h[None, :] < H, 0
    ).to(tl.float32)
    for j in tl.static_range(M):
        comb = tl.load(COMB + t * M * M + i * M + j)
        post = tl.load(POST + t * M + j)
        out = post * x + tl.sum(comb[:, None] * residual, 0)
        tl.store(OUT + t * M * H + j * H + h, out, h < H)


def hc_pre(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult_value=2.0,
    hc_norm_weight=None,
    out_norm_weight=None,
    out_norm_eps=None,
):
    tokens, total = x.shape
    hidden = total // hc_mult
    layer_input = x.new_empty((tokens, hidden))
    comb = torch.empty(
        (tokens, hc_mult * hc_mult), device=x.device, dtype=torch.float32
    )
    post = torch.empty((tokens, hc_mult), device=x.device, dtype=torch.float32)
    if tokens:
        pre = torch.empty_like(post)
        mixes = F.linear(x.float(), hc_fn)
        _hc_coefficients[(tokens,)](
            x,
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            TOTAL=total,
            BT=triton.next_power_of_2(total),
            M=hc_mult,
            RMS_EPS=rms_eps,
            EPS=hc_eps,
            ITERS=sinkhorn_iters,
            POST_MULT=post_mult_value,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
        _hc_pre_apply[(tokens, triton.cdiv(hidden, 1024))](
            x,
            pre,
            layer_input,
            H=hidden,
            M=hc_mult,
            B=1024,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
    return layer_input, comb, post, False


def hc_post(x, residual, h_post, h_res, hc_mult):
    tokens, hidden = x.shape
    out = x.new_empty((tokens, hc_mult * hidden))
    if tokens:
        _hc_post_apply[(tokens, triton.cdiv(hidden, 512))](
            x,
            residual,
            h_post,
            h_res,
            out,
            H=hidden,
            M=hc_mult,
            B=512,
            num_warps=1,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
    return out
