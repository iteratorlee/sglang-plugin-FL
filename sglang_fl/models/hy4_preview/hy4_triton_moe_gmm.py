"""Triton grouped-expert GMM for HYV4 MoE (Ascend, graph-safe).

Replaces torch.ops.npu.npu_grouped_matmul inside the captured graph. The NPU
op produced non-reference values under NPUGraph replay in the component probe,
while these plain Triton launches captured and replayed value-exactly.

Weights are pre-dequantized bf16 ND [E, N, K] (out, in). Tokens are pre-sorted
by expert via npu_moe_init_routing, so expert e occupies rows
[off[e], off[e]+cnt[e]) of x.

Layout contract:
  x      [T, K]      bf16 routed hidden (sorted by expert)
  weight [E, N, K]   bf16 dequantized expert weights (row-major)
  cnt    [E]         int64 tokens per expert
  off    [E]         int64 start row of each expert
  out    [T, N]      bf16
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _hy4_expert_gmm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    cnt_ptr,
    off_ptr,
    T,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    e = tl.program_id(0)
    n_tile = tl.program_id(1)

    n0 = n_tile * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    n_e = tl.load(cnt_ptr + e)
    off_e = tl.load(off_ptr + e)
    w_base = w_ptr + e.to(tl.int64) * N * K

    for m0 in range(0, n_e, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        rows = off_e + offs_m
        # The routing metadata is produced at runtime and is captured by
        # NPUGraph.  Keep every memory access inside the actual routed-token
        # buffer even if a bad count/offset pair reaches the kernel; checking
        # only ``offs_m < n_e`` can otherwise corrupt the next operator's
        # workspace before the asynchronous error is reported.
        m_mask = (offs_m < n_e) & (rows >= 0) & (rows < T)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            xb = tl.load(
                x_ptr + rows[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            # weight stored [E, N, K] row-major: element (n, k) at n*K + k.
            # Load the [K, N] tile directly so tl.dot(xb, wb) needs no tl.trans.
            wb = tl.load(
                w_base + offs_n[None, :] * K + offs_k[:, None],
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            acc += tl.dot(xb, wb)
        tl.store(
            out_ptr + rows[:, None] * N + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )


_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_K = 64


def hy4_expert_gmm(x, weight, cnt):
    """Grouped expert matmul: out[e_rows] = x[e_rows] @ weight[e].T

    x: [T, K] bf16 (routed, sorted by expert)
    weight: [E, N, K] bf16 dequantized
    cnt: [E] int64 tokens per expert (sorted ascending by expert id)
    returns [T, N] bf16
    """
    E = weight.shape[0]
    N = weight.shape[1]  # weight is [E, N, K]
    K = weight.shape[2]
    T = x.shape[0]
    device = x.device

    off = torch.cumsum(cnt, dim=0) - cnt
    if cnt.ndim != 1 or cnt.numel() != E:
        raise RuntimeError(
            f"HYV4 expert counts must have shape ({E},), got {tuple(cnt.shape)}"
        )
    out = torch.zeros((T, N), dtype=torch.bfloat16, device=device)

    grid = (E, triton.cdiv(N, _BLOCK_N))
    _hy4_expert_gmm_kernel[grid](
        x, weight, out, cnt, off, T, N, K,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    return out


@triton.jit
def _hy4_expert_gmm_i8_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    token_scale_ptr,
    out_ptr,
    cnt_ptr,
    off_ptr,
    T,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_TOKEN_SCALE: tl.constexpr,
):
    e = tl.program_id(0)
    n_tile = tl.program_id(1)

    n0 = n_tile * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    n_e = tl.load(cnt_ptr + e)
    off_e = tl.load(off_ptr + e)
    w_base = w_ptr + e.to(tl.int64) * N * K
    # per-output-row scale [E, N]; hoisted out of the m loop.
    s = tl.load(s_ptr + e * N + offs_n, mask=n_mask, other=0.0)

    for m0 in range(0, n_e, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        rows = off_e + offs_m
        m_mask = (offs_m < n_e) & (rows >= 0) & (rows < T)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            xb = tl.load(
                x_ptr + rows[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            # int8 weights convert to bf16 exactly (7 < 8 mantissa bits).
            wb = tl.load(
                w_base + offs_n[None, :] * K + offs_k[:, None],
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            acc += tl.dot(xb, wb)
        if HAS_TOKEN_SCALE:
            # A8 values, like INT8 weights, convert exactly to BF16. Keep the
            # per-token scale in FP32 until after accumulation: reconstructing
            # BF16 activations before dot introduces an extra rounding at
            # every element that is not part of the W8A8 contract.
            token_scale = tl.load(
                token_scale_ptr + rows, mask=m_mask, other=0.0
            ).to(tl.float32)
            acc = acc * token_scale[:, None]
        acc = acc * s[None, :].to(tl.float32)
        tl.store(
            out_ptr + rows[:, None] * N + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _hy4_expert_gmm_small_rows_kernel(
    x_ptr, w_ptr, s_ptr, token_scale_ptr, out_ptr, cnt_ptr, off_ptr,
    N, K, E: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_TOKEN_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(cnt_ptr + experts, mask=experts < E, other=0)
    starts = tl.load(off_ptr + experts, mask=experts < E, other=0)
    owns_row = (experts < E) & (starts <= row) & (row < starts + counts)
    expert = tl.max(tl.where(owns_row, experts, -1), axis=0)
    valid = expert >= 0
    expert = tl.maximum(expert, 0)
    # Only one real row: all other M lanes are padding for the matrix engine.
    lanes = tl.arange(0, BLOCK_M)
    row_mask = (lanes == 0) & valid
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + (row + lanes[:, None]) * K + offs_k[None, :],
            mask=row_mask[:, None] & (offs_k[None, :] < K), other=0,
        ).to(tl.bfloat16)
        w = tl.load(
            w_ptr + expert.to(tl.int64) * N * K
            + offs_n[None, :] * K + offs_k[:, None],
            mask=valid & (offs_n[None, :] < N) & (offs_k[:, None] < K), other=0,
        ).to(tl.bfloat16)
        acc += tl.dot(x, w)
    if HAS_TOKEN_SCALE:
        token_scale = tl.load(token_scale_ptr + row).to(tl.float32)
        acc *= token_scale
    scale = tl.load(s_ptr + expert * N + offs_n, mask=valid & (offs_n < N), other=0)
    acc *= scale[None, :].to(tl.float32)
    tl.store(
        out_ptr + (row + lanes[:, None]) * N + offs_n[None, :],
        acc.to(tl.bfloat16), mask=(lanes[:, None] == 0) & (offs_n[None, :] < N),
    )


@triton.jit
def _hy4_gemv_split_kernel(
    x_ptr, w_ptr, partial_ptr, cnt_ptr, off_ptr,
    N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    row, n_tile, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(cnt_ptr + experts, mask=experts < E, other=0)
    starts = tl.load(off_ptr + experts, mask=experts < E, other=0)
    owns_row = (experts < E) & (starts <= row) & (row < starts + counts)
    expert = tl.max(tl.where(owns_row, experts, -1), axis=0)
    valid = expert >= 0
    expert = tl.maximum(expert, 0)
    ns = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = split * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + row * K + ks, mask=ks < K, other=0).to(tl.int32)
    w = tl.load(
        w_ptr + expert.to(tl.int64) * N * K + ns[:, None] * K + ks[None, :],
        mask=valid & (ns[:, None] < N) & (ks[None, :] < K), other=0,
    ).to(tl.int32)
    sums = tl.sum(w * x[None, :], axis=1)
    tl.store(partial_ptr + (row * SPLITS + split) * N + ns, sums, mask=ns < N)


@triton.jit
def _hy4_gemv_finalize_kernel(
    partial_ptr, token_scale_ptr, weight_scale_ptr, out_ptr, cnt_ptr, off_ptr,
    N: tl.constexpr, E: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_SPLITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    experts = tl.arange(0, BLOCK_E)
    counts = tl.load(cnt_ptr + experts, mask=experts < E, other=0)
    starts = tl.load(off_ptr + experts, mask=experts < E, other=0)
    owns_row = (experts < E) & (starts <= row) & (row < starts + counts)
    expert = tl.max(tl.where(owns_row, experts, -1), axis=0)
    valid = expert >= 0
    expert = tl.maximum(expert, 0)
    ns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    splits = tl.arange(0, BLOCK_SPLITS)
    partial = tl.load(
        partial_ptr + (row * SPLITS + splits[:, None]) * N + ns[None, :],
        mask=(splits[:, None] < SPLITS) & (ns[None, :] < N), other=0,
    )
    acc = tl.sum(partial, axis=0).to(tl.float32)
    token_scale = tl.load(token_scale_ptr + row).to(tl.float32)
    weight_scale = tl.load(
        weight_scale_ptr + expert * N + ns, mask=valid & (ns < N), other=0,
    ).to(tl.float32)
    out = (acc * token_scale) * weight_scale
    tl.store(out_ptr + row * N + ns, out.to(tl.bfloat16), mask=ns < N)


def hy4_expert_gmm_i8(x, weight, scale, cnt, pertoken_scale=None):
    """Grouped expert matmul with int8 weights and per-output-row scale.

    Computes out[t, n] = scale[e, n] * sum_k x[t, k] * weight[e, n, k].
    Keeps weights in int8 ND (half the memory of a bf16 dequant cache) and
    applies the per-row scale after the fp32 accumulation.

    x: [T, K] bf16, or int8 when pertoken_scale is supplied
    weight: [E, N, K] int8 ND
    scale: [E, N] bf16 per-output-row scale
    cnt: [E] int64 tokens per expert (sorted ascending by expert id)
    pertoken_scale: optional [T] FP32 dynamic A8 scale, applied after dot
    returns [T, N] bf16
    """
    E = weight.shape[0]
    N = weight.shape[1]
    K = weight.shape[2]
    T = x.shape[0]
    device = x.device

    if pertoken_scale is not None:
        if x.dtype != torch.int8 or pertoken_scale.dtype != torch.float32:
            raise RuntimeError("HYV4 A8 GMM requires INT8 input and FP32 token scales")
        if pertoken_scale.numel() != T or not pertoken_scale.is_contiguous():
            raise RuntimeError("HYV4 A8 GMM requires one contiguous scale per row")

    off = torch.cumsum(cnt, dim=0) - cnt
    if cnt.ndim != 1 or cnt.numel() != E:
        raise RuntimeError(
            f"HYV4 expert counts must have shape ({E},), got {tuple(cnt.shape)}"
        )
    out = torch.zeros((T, N), dtype=torch.bfloat16, device=device)

    if T <= 16 and pertoken_scale is not None and K >= 512:
        # A BS1 MoE is eight independent GEMVs, not a large GEMM. INT32 vector
        # products/reductions preserve the native A8 contract. Split-K exposes
        # parallelism without 96 serial padded matrix dot steps for K=6144.
        block_k = min(512, triton.next_power_of_2(K))
        splits = triton.cdiv(K, block_k)
        partial = torch.empty((T, splits, N), dtype=torch.int32, device=device)
        _hy4_gemv_split_kernel[(T, triton.cdiv(N, 8), splits)](
            x, weight, partial, cnt, off, N=N, K=K, E=E, SPLITS=splits,
            BLOCK_E=triton.next_power_of_2(E), BLOCK_N=8, BLOCK_K=block_k,
            num_warps=4,
        )
        _hy4_gemv_finalize_kernel[(T, triton.cdiv(N, 64))](
            partial, pertoken_scale, scale, out, cnt, off,
            N=N, E=E, SPLITS=splits, BLOCK_E=triton.next_power_of_2(E),
            BLOCK_SPLITS=triton.next_power_of_2(splits), BLOCK_N=64,
            num_warps=4,
        )
        return out

    # TP32/BS1 has only eight routed rows. Scheduling all 256 experts creates
    # thousands of empty programs (especially the 6144-wide down projection).
    # Resolve each row's expert from device counts during every graph replay.
    if T <= 16:
        _hy4_expert_gmm_small_rows_kernel[(T, triton.cdiv(N, _BLOCK_N))](
            x, weight, scale, pertoken_scale, out, cnt, off,
            N=N, K=K, E=E, BLOCK_E=triton.next_power_of_2(E),
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
            HAS_TOKEN_SCALE=pertoken_scale is not None, num_warps=4,
        )
        return out

    grid = (E, triton.cdiv(N, _BLOCK_N))
    _hy4_expert_gmm_i8_kernel[grid](
        x, weight, scale, pertoken_scale, out, cnt, off, T, N, K,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        HAS_TOKEN_SCALE=pertoken_scale is not None,
        num_warps=4,
    )
    return out
