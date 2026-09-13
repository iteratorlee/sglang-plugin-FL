"""Transactional convolution and KDA state for linear, top-k-one MTP verify.

Verify writes per-token scratch state; only the accepted prefix is committed.
The persistent caches remain unchanged while draft tokens are being verified.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_verify(X, STATE, W, IDS, STARTS, OUT, MID,
                 DIM: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    slot = tl.load(IDS + request)
    if slot > 0:
        d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        begin = tl.load(STARTS + request)
        end = tl.load(STARTS + request + 1)
        s0 = tl.load(STATE + slot * DIM * 3 + d * 3, d < DIM, 0)
        s1 = tl.load(STATE + slot * DIM * 3 + d * 3 + 1, d < DIM, 0)
        s2 = tl.load(STATE + slot * DIM * 3 + d * 3 + 2, d < DIM, 0)
        w0 = tl.load(W + d * 4, d < DIM, 0).to(tl.float32)
        w1 = tl.load(W + d * 4 + 1, d < DIM, 0).to(tl.float32)
        w2 = tl.load(W + d * 4 + 2, d < DIM, 0).to(tl.float32)
        w3 = tl.load(W + d * 4 + 3, d < DIM, 0).to(tl.float32)
        for i in range(end - begin):
            x = tl.load(X + (begin + i) * DIM + d, d < DIM, 0)
            y = ((s0.to(tl.float32) * w0 + s1.to(tl.float32) * w1)
                 + s2.to(tl.float32) * w2) + x.to(tl.float32) * w3
            y = y / (1 + tl.exp(-y))
            tl.store(OUT + (begin + i) * DIM + d, y, d < DIM)
            mid = MID + ((request * STEPS + i) * DIM + d) * 3
            tl.store(mid, s1, d < DIM)
            tl.store(mid + 1, s2, d < DIM)
            tl.store(mid + 2, x, d < DIM)
            s0 = s1
            s1 = s2
            s2 = x


def conv_verify(x, state, weight, indices, starts, intermediate):
    dim = x.shape[-1]
    if (x.ndim != 2 or state.shape[1:] != (dim, 3)
        or weight.shape != (dim, 4) or indices.numel() + 1 != starts.numel()
        or intermediate.ndim != 4 or intermediate.shape[2:] != (dim, 3)
        or intermediate.shape[0] < indices.numel()
        or x.dtype != torch.bfloat16 or state.dtype != x.dtype
        or intermediate.dtype != state.dtype
        or not all(t.is_contiguous() for t in (x, state, weight, indices, starts, intermediate))):
        raise ValueError("Invalid GLM MTP convolution layout: " + str(
            [(tuple(t.shape), str(t.dtype), t.stride()) for t in
             (x, state, weight, indices, starts, intermediate)]))
    out = torch.zeros_like(x)
    _conv_verify[(indices.numel(), triton.cdiv(dim, 128))](
        x, state, weight, indices, starts, out, intermediate,
        DIM=dim, STEPS=intermediate.shape[1], BLOCK=128,
        enable_fp_fusion=False, enable_auto_bind_sub_block=False)
    return out


@triton.jit
def _commit_state(DST, SRC, IDS, ACCEPTED,
                  DST_LAYER_STRIDE: tl.constexpr, SRC_LAYER_STRIDE: tl.constexpr,
                  ELEMENTS: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    layer = tl.program_id(1)
    slot = tl.load(IDS + request)
    accepted = tl.load(ACCEPTED + request)
    if (slot > 0) & (accepted >= 0) & (accepted < STEPS):
        e = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(SRC + layer * SRC_LAYER_STRIDE
                        + (request * STEPS + accepted) * ELEMENTS + e,
                        e < ELEMENTS, 0)
        tl.store(DST + layer * DST_LAYER_STRIDE + slot * ELEMENTS + e,
                 value, e < ELEMENTS)


def commit_state(destination, intermediate, indices, accepted_steps):
    """Accepted steps are zero-based: zero commits the verified root token."""
    layers, pool = destination.shape[:2]
    elements = destination[0, 0].numel()
    if (intermediate.shape[0] != layers
        or intermediate.shape[1] < indices.numel()
        or intermediate[0, 0, 0].numel() != elements
        or indices.shape != accepted_steps.shape
        or destination.dtype != intermediate.dtype
        or not destination.is_contiguous() or not intermediate.is_contiguous()):
        raise ValueError("Invalid GLM MTP state commit layout")
    if indices.numel():
        _commit_state[(indices.numel(), layers, triton.cdiv(elements, 1024))](
            destination, intermediate, indices, accepted_steps,
            DST_LAYER_STRIDE=destination.stride(0),
            SRC_LAYER_STRIDE=intermediate.stride(0),
            ELEMENTS=elements, STEPS=intermediate.shape[2], BLOCK=1024,
            enable_auto_bind_sub_block=False)
