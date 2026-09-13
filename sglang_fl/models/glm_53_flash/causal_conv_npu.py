"""Single-token Ascend KDA convolution with graph-safe rolling state."""

import torch
import triton
import triton.language as tl


@triton.jit
def _glm_causal_conv_step(
    X, STATE, WEIGHT, INDICES, DIM: tl.constexpr, BLOCK: tl.constexpr
):
    token = tl.program_id(0)
    slot = tl.load(INDICES + token)
    # Slot zero belongs to graph padding. Neither input nor cache is modified.
    if slot != 0:
        d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        k = tl.arange(0, 4)
        history = tl.load(
            STATE + slot * DIM * 3 + d[:, None] * 3 + k[None, :],
            (d[:, None] < DIM) & (k[None, :] < 3),
            0,
        )
        x = tl.load(X + token * DIM + d, d < DIM, 0)
        values = tl.where(k[None, :] < 3, history, x[:, None])
        weight = tl.load(WEIGHT + d[:, None] * 4 + k[None, :], d[:, None] < DIM, 0).to(
            tl.float32
        )
        products = values.to(tl.float32) * weight
        a0 = tl.sum(tl.where(k[None, :] == 0, products, 0), 1)
        a1 = tl.sum(tl.where(k[None, :] == 1, products, 0), 1)
        a2 = tl.sum(tl.where(k[None, :] == 2, products, 0), 1)
        a3 = tl.sum(tl.where(k[None, :] == 3, products, 0), 1)
        acc = ((a0 + a1) + a2) + a3
        output = acc / (1 + tl.exp(-acc))
        shifted = tl.gather(
            values,
            tl.broadcast_to(tl.minimum(k + 1, 3)[None, :], (BLOCK, 4)),
            1,
        )
        tl.store(
            STATE + slot * DIM * 3 + d[:, None] * 3 + k[None, :],
            shifted,
            (d[:, None] < DIM) & (k[None, :] < 3),
        )
        tl.store(X + token * DIM + d, output, d < DIM)


def supports_causal_conv_step(x, state, weight, bias, indices):
    """Restrict the fast path to the tested attention-TP layouts and one token/slot."""
    return (
        bias is None
        and x.device.type == "npu"
        and all(t.device == x.device for t in (state, weight, indices))
        and x.dtype == state.dtype == torch.bfloat16
        and weight.dtype in (torch.float32, torch.bfloat16)
        and indices.dtype == torch.int32
        and x.ndim == 2
        and x.shape[1] in (768, 1536, 3072, 6144)
        and state.ndim == 3
        and state.shape[1:] == (x.shape[1], 3)
        and weight.shape == (x.shape[1], 4)
        and indices.shape == (x.shape[0],)
        and all(t.is_contiguous() for t in (x, state, weight, indices))
    )


def causal_conv_step(x, state, weight, indices):
    """Update contiguous BF16 input and cache in place; weight multiply is FP32."""
    if x.shape[0]:
        _glm_causal_conv_step[(x.shape[0], triton.cdiv(x.shape[1], 32))](
            x,
            state,
            weight,
            indices,
            DIM=x.shape[1],
            BLOCK=32,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
        )
    return x
