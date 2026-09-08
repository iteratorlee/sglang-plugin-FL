"""Triton batched sink-decode attention for HYV4's dense MLA (Ascend).

Replaces the npu_fusion_attention + dynamic-gather decode path with a single
kernel whose inputs are all device tensors with stable addresses, so the decode
step is capturable by SGLang's CUDA graph machinery:

- per-request KV rows are resolved inside the kernel from req_to_token (no
  host-side variable-length gather, no Python lists),
- the learnable sink parameter is folded into the online softmax,
- query/KV consumption and output production stay in one graph-safe launch.

The query is passed as TWO separate tensors (q_nope and q_rope) and the
concatenation happens inside the kernel.  This is deliberate: ``torch.cat``
compiles to an ``rtMemcpy`` on Ascend, and this CANN build rejects rtMemcpy
inside CUDA-graph capture (``the current capture mode does not support this
operation``), which corrupts the captured graph for batch sizes >= 24.
Keeping every data move inside the Triton kernel makes the captured region
pure kernel launches.

Layout contract (matches the NPUMLA token pool):
  key_cache   [pool, kv_heads, kv_lora_rank]        (k_nope)
  rope_cache  [pool, kv_heads, qk_rope_head_dim]    (k_pe)
  q_nope      [S, q_heads, kv_lora_rank]            (query nope part)
  q_rope      [S, q_heads, qk_rope_head_dim]        (query rope part)
  req_to_token [max_reqs, max_ctx]                   token id of (req, pos)
  A token id t addresses row t of key_cache/rope_cache, i.e. byte offset
  t * (kv_heads * dim) inside the flattened [pool, kv_heads, dim] buffer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hy4_mla_decode_sinks_kernel(
    q_nope,
    q_rope,
    key_cache,
    rope_cache,
    sinks,
    attn_out,
    req_to_token,
    req_pool_indices,
    seq_lens,
    scale,
    req_stride,
    qn_stride_s,
    qn_stride_h,
    qn_stride_d,
    qr_stride_s,
    qr_stride_h,
    qr_stride_d,
    key_stride_t,
    key_stride_h,
    key_stride_d,
    rope_stride_t,
    rope_stride_h,
    rope_stride_d,
    out_stride_s,
    out_stride_h,
    out_stride_d,
    D_N: tl.constexpr,
    D_R: tl.constexpr,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    BR: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_gh = tl.program_id(1)
    i_kvh = i_gh * BR // (Q_HEADS // K_HEADS)

    kv_len = tl.load(seq_lens + i_s)
    req = tl.load(req_pool_indices + i_s)

    h = i_gh * BR + tl.arange(0, BR)
    off_d = tl.arange(0, D_N)
    off_dr = tl.arange(0, D_R)

    # q_rope is commonly a non-contiguous split view of [S,H,D_N+D_R]; use
    # both query tensors' logical strides instead of assuming packed storage.
    q_n = tl.load(
        q_nope
        + i_s * qn_stride_s
        + h[:, None] * qn_stride_h
        + off_d[None, :] * qn_stride_d
    ).to(tl.float32)
    q_r = tl.load(
        q_rope
        + i_s * qr_stride_s
        + h[:, None] * qr_stride_h
        + off_dr[None, :] * qr_stride_d
    ).to(tl.float32)

    sink = tl.load(sinks + h).to(tl.float32)
    m_i = tl.zeros([BR], dtype=tl.float32) + sink
    l_i = tl.zeros([BR], dtype=tl.float32)
    acc = tl.zeros([BR, D_N], dtype=tl.float32)

    for pos0 in range(0, kv_len, BR):
        pos = pos0 + tl.arange(0, BR)
        valid = pos < kv_len
        tid = tl.load(
            req_to_token + req * req_stride + pos, mask=valid, other=0
        )

        k_nope = tl.load(
            key_cache
            + tid[:, None] * key_stride_t
            + i_kvh * key_stride_h
            + off_d[None, :] * key_stride_d,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        k_rope = tl.load(
            rope_cache
            + tid[:, None] * rope_stride_t
            + i_kvh * rope_stride_h
            + off_dr[None, :] * rope_stride_d,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # qk = (q_nope . k_nope) + (q_rope . k_rope), split because the two KV
        # halves live in different pool buffers.
        qk = tl.sum(q_n[:, None, :] * k_nope[None, :, :], axis=2) + tl.sum(
            q_r[:, None, :] * k_rope[None, :, :], axis=2
        )
        qk = qk * scale
        qk = tl.where(valid[None, :], qk, float("-inf"))

        # Online softmax with the per-head learnable sink folded in.
        m_new = tl.maximum(tl.max(qk, 1), m_i)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.sum(
            p[:, :, None] * k_nope[None, :, :], axis=1
        )
        m_i = m_new

    sink_v = tl.exp(sink - m_i)
    l_i = l_i + sink_v
    acc = acc / l_i[:, None]
    tl.store(
        attn_out
        + i_s * out_stride_s
        + h[:, None] * out_stride_h
        + off_d[None, :] * out_stride_d,
        acc.to(attn_out.dtype.element_ty),
    )


def hy4_mla_decode_sinks_triton(
    q_nope,
    q_rope,
    key_cache,
    rope_cache,
    sinks,
    req_to_token,
    req_pool_indices,
    seq_lens,
    scale,
    q_head_num,
    kv_lora_rank,
    qk_rope_head_dim,
):
    """Run the batched sink decode for one layer.

    ``q_nope`` is ``[S, q_head_num, kv_lora_rank]`` and ``q_rope`` is
    ``[S, q_head_num, qk_rope_head_dim]``; both are passed untouched and their
    full logical strides are consumed by the kernel. Returns ``[S, q_head_num,
    kv_lora_rank]`` attention output for the current decode step (all ``S``
    requests share the same query step).
    """
    S = q_nope.shape[0]

    if key_cache.ndim == 4:
        key_cache = key_cache.flatten(0, 1)  # [pool, K_HEADS, D_N]
        rope_cache = rope_cache.flatten(0, 1)
    k_head_num = key_cache.shape[1]
    d_n = key_cache.shape[2]
    d_r = rope_cache.shape[2]

    br = min(q_head_num // k_head_num, 16)
    group_num = q_head_num // br

    attn_out = torch.zeros(
        (S, q_head_num, d_n), dtype=q_nope.dtype, device=q_nope.device
    )
    grid = [S, group_num]
    _hy4_mla_decode_sinks_kernel[grid](
        q_nope,
        q_rope,
        key_cache,
        rope_cache,
        sinks,
        attn_out,
        req_to_token,
        req_pool_indices,
        seq_lens,
        scale,
        req_to_token.stride(0),
        *q_nope.stride(),
        *q_rope.stride(),
        *key_cache.stride(),
        *rope_cache.stride(),
        *attn_out.stride(),
        d_n,
        d_r,
        q_head_num,
        k_head_num,
        br,
    )
    return attn_out


@triton.jit
def _hy4_kv_scatter_kernel(
    pool_ptr,
    data_ptr,
    loc_ptr,
    seq_lens_ptr,
    D,
    data_stride_s,
    data_stride_d,
    BLOCK: tl.constexpr,
):
    """Write data row s (length D) into pool row loc[s].

    Replaces torch_npu.npu_scatter_nd_update_ inside the captured decode graph:
    that NPU op bakes the index tensor at capture time, so on replay it writes
    to the stale first-step slots and the KV cache never accumulates (decode
    gets stuck on a repeated token).  A plain triton store re-reads ``loc``
    from device memory at every replay, so it is value-exact like the rest of
    the triton path.
    """
    s = tl.program_id(0)
    db = tl.program_id(1)
    t = tl.load(loc_ptr + s).to(tl.int64)
    active = tl.load(seq_lens_ptr + s) > 0
    d0 = db * BLOCK + tl.arange(0, BLOCK)
    dm = d0 < D
    v = tl.load(
        data_ptr
        + s.to(tl.int64) * data_stride_s
        + d0 * data_stride_d,
        mask=dm,
        other=0.0,
    )
    # Graph padding rows use seq_len=0 and loc=0. They must not race to
    # overwrite slot 0 in the real KV pool.
    tl.store(pool_ptr + t * D + d0, v, mask=dm & active)


_BLOCK_SCATTER = 128


def hy4_kv_scatter(pool, data, loc, seq_lens):
    """Scatter data [S, D] rows into the KV pool at token rows loc [S].

    ``pool`` is SGLang's contiguous layer key/value cache buffer; its leading
    dimensions may vary, but flattening must remain a view of the same storage.
    ``data`` is the per-token latent/rope chunk [S, D].
    Rows whose graph-static ``seq_lens`` entry is zero are padding and do not
    write. Graph-safe: identical addresses across replays, contents re-read.
    """
    S, D = data.reshape(-1, data.shape[-1]).shape
    data2 = data.reshape(-1, D)
    pool_flat = pool.reshape(-1)
    grid = (S, triton.cdiv(D, _BLOCK_SCATTER))
    _hy4_kv_scatter_kernel[grid](
        pool_flat,
        data2,
        loc,
        seq_lens,
        D,
        data2.stride(0),
        data2.stride(1),
        BLOCK=_BLOCK_SCATTER,
    )
    return pool
