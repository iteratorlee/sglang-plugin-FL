"""Graph-dynamic KV/index cache writes for HYV4 native DSA on Ascend."""

import triton
import triton.language as tl


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
    data2 = data.reshape(-1, data.shape[-1])
    S, D = data2.shape
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
