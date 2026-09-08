"""Graph-safe PyTorch mHC reference used on Ascend.

The implementation follows the model vendor's equations.  It deliberately
uses regular PyTorch operators so torch-npu can capture it in an NPU graph.
"""

import torch
import torch.nn.functional as F


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.repeat(1, n)


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.unflatten(-1, (n, -1)).mean(dim=-2)


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult_value: float = 2.0,
    hc_norm_weight=None,
    out_norm_weight=None,
    out_norm_eps=None,
):
    del hc_norm_weight, out_norm_weight, out_norm_eps
    tokens, total = x.shape
    hidden = total // hc_mult
    if tokens == 0:
        return (
            x.new_empty((0, hidden)),
            torch.empty((0, hc_mult * hc_mult), device=x.device, dtype=torch.float32),
            torch.empty((0, hc_mult), device=x.device, dtype=torch.float32),
            False,
        )
    residual = x.reshape(tokens, hc_mult, hidden)
    flat = residual.reshape(tokens, total).float()
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, hc_fn) * inv_rms
    pre_raw = mixes[:, :hc_mult]
    post_raw = mixes[:, hc_mult : 2 * hc_mult]
    comb_raw = mixes[:, 2 * hc_mult :].reshape(tokens, hc_mult, hc_mult)
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + hc_eps
    post = post_mult_value * torch.sigmoid(
        post_raw * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = comb_raw * hc_scale[2] + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(x.dtype)
    return layer_input, comb.reshape(tokens, -1), post, False


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    hc_mult: int,
) -> torch.Tensor:
    tokens, hidden = x.shape
    if tokens == 0:
        return x.new_empty((0, hc_mult * hidden))
    residual = residual.reshape(tokens, hc_mult, hidden)
    comb = h_res.reshape(tokens, hc_mult, hc_mult)
    post = h_post.reshape(tokens, hc_mult, 1)
    out = post * x.unsqueeze(1) + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(
        dim=1
    )
    return out.to(x.dtype).reshape(tokens, -1)
