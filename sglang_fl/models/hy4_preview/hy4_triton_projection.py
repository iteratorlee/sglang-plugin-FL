"""Memory-efficient Triton projections for HYV4 MLA on Ascend.

The input layout is [tokens, heads, K] and weights are [heads, K, N].
The kernel consumes those layouts directly and writes [tokens, heads, N],
avoiding the two full-size transpose/contiguous buffers created by torch.bmm.
The optional per-token gate is fused into the value projection store.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hy4_head_bmm_kernel(
    a_ptr,
    w_ptr,
    gate_ptr,
    out_ptr,
    T,
    stride_at,
    stride_ah,
    stride_ak,
    stride_wh,
    stride_wk,
    stride_wn,
    stride_gt,
    stride_gh,
    stride_gn,
    H: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    HAS_GATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < T
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr
            + offs_m[:, None] * stride_at
            + head * stride_ah
            + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        w = tl.load(
            w_ptr
            + head * stride_wh
            + offs_k[:, None] * stride_wk
            + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(a, w)

    if HAS_GATE:
        # HYV4's attention gate is elementwise [T, H, N], not a scalar per
        # token.  Preserve both the local-head and output-channel offsets.
        gate = tl.load(
            gate_ptr
            + offs_m[:, None] * stride_gt
            + head * stride_gh
            + offs_n[None, :] * stride_gn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc *= gate

    tl.store(
        out_ptr + offs_m[:, None] * (H * N) + head * N + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def hy4_head_bmm(a, weight, gate=None):
    """Compute out[t,h,n] = sum_k a[t,h,k] * weight[h,k,n]."""
    assert a.ndim == 3 and weight.ndim == 3
    t, h, k = a.shape
    assert weight.shape[0] == h and weight.shape[1] == k
    n = weight.shape[2]
    if gate is not None:
        assert gate.numel() == t * h * n
        gate_view = gate.view(t, h, n)
        gate_strides = gate_view.stride()
    else:
        gate_view = a
        gate_strides = (0, 0, 0)
    block_m, block_n, block_k = 32, 64, 64
    out = torch.empty((t, h, n), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(t, block_m), h, triton.cdiv(n, block_n))
    _hy4_head_bmm_kernel[grid](
        a,
        weight,
        gate_view,
        out,
        t,
        *a.stride(),
        *weight.stride(),
        *gate_strides,
        H=h,
        K=k,
        N=n,
        HAS_GATE=gate is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def hy4_head_bmm_prefill(a, weight, gate=None):
    """Use the native eager batched matmul for variable-token prefill.

    On this Triton/Ascend stack, the second partial ``BLOCK_M`` program of
    ``hy4_head_bmm`` can overflow for real model activations (the first failing
    case is token 32 of a 37-token prefill).  Prefill is outside NPUGraph, so a
    contiguous native batched matmul avoids that partial-tile kernel path.
    Decode keeps using ``hy4_head_bmm`` because its graph batch is fixed to 1.
    """
    assert a.ndim == 3 and weight.ndim == 3
    t, h, k = a.shape
    assert weight.shape[0] == h and weight.shape[1] == k
    n = weight.shape[2]
    out = torch.bmm(
        a.transpose(0, 1).contiguous(), weight.contiguous()
    ).transpose(0, 1).contiguous()
    if gate is not None:
        assert gate.numel() == t * h * n
        out.mul_(gate.view(t, h, n))
    return out


@triton.jit
def _hy4_clamped_swiglu_kernel(x_ptr, out_ptr, rows, width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = (row < rows) & (offs < width)
    base = row * (2 * width)
    gate = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + base + width + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.maximum(tl.minimum(up, 10.0), -10.0)
    act = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(out_ptr + row * width + offs, act.to(tl.bfloat16), mask=mask)


def hy4_clamped_swiglu(x):
    """Fuse split, both clamps, SiLU and multiply for [T, 2N] input."""
    rows, twice_width = x.shape
    width = twice_width // 2
    if rows != 8:
        # Only the TP32/BS1 top-8 decode shape has passed the Triton kernel's
        # value/graph checks on this compiler. At the actual prefill widths,
        # routed T=296/8192 silently returns incorrect values. Prefill is not
        # captured: compute the trained clamps and activation in native FP32,
        # with a single rounding to the model dtype after the multiply.
        gate = x[:, :width].float().clamp(max=10.0)
        up = x[:, width:].float().clamp(min=-10.0, max=10.0)
        return (torch.nn.functional.silu(gate) * up).to(x.dtype)
    out = torch.empty((rows, width), dtype=x.dtype, device=x.device)
    block = 256
    _hy4_clamped_swiglu_kernel[(rows, triton.cdiv(width, block))](
        x, out, rows, width, BLOCK=block, num_warps=4
    )
    return out
