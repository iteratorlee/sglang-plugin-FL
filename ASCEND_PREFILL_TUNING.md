# Ascend Qwen3.6 prefill tuning

These plugin patches target Qwen3.6-27B TP=2 BF16 prefill. All are **off by
default**. For **27B only**, set these before starting new workers:

```bash
export SGLANG_FL_FIA_TND_GQA=1
export SGLANG_FL_GDN_GATING_BLOCK24=1
export SGLANG_FL_PREFILL_NZ_GATE_UP=1
```

For **35B-A3B**, leave all three unset or `0`: its shapes do not benefit.
Keep the existing plugin/FlagGems configuration and ordinary decode graphs.
To roll back, unset these options and restart workers; no weight or source
restoration is needed. SGLang, FlagGems and `sgl_kernel_npu` are not modified.

## Scope and compatibility

Validated on Ascend 910C with image
`quay.io/ascend/sglang:v0.5.11-cann8.5.0-a3` and **installed** `sgl_kernel_npu`
2026.3.1, not its upstream checkout. Revalidate other software versions.
Measurements used TP=2, C=64, output 1024, prefill chunks 16384 and memory
fraction 0.82. MTP/speculation, prefix/radix cache and piecewise graphs were off.

| Option | Guarded shape | Change |
|---|---|---|
| FIA TND GQA | One fresh, unpadded 16384-token EXTEND, Q12/K2/V2/D256, BF16 | Use TND and return its output directly. Preserve causal mask/scale and the single KV write before attention. Accept contiguous V or zero-copy packed stride `(7168,256,1)`. |
| GDN gating BLOCK24 | Contiguous BF16 a/b `[16384,24]`, FP32 A_log, FP32/BF16 dt_bias | Reuse the installed kernel with one 24-head group instead of three groups of eight. Keep FP32 outputs, beta and threshold. |
| Ephemeral NZ gate-up | Contiguous BF16 x `[16384,5120]`, ND weight `[17408,5120]`, exact unquantized method, no bias | Convert the current weight to temporary FRACTAL_NZ for linear. Preserve the original ND Parameter/storage, with no persistent or stale-weight cache. |

Unverified shapes, dtypes, strides and attention modes fall back. Decode uses
the original implementations. NZ also falls back during graph capture because
CANN's ND-to-NZ Identity is not capturable; `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`
prevents its installation. Installation is idempotent and checks function/kernel
signatures, but cannot guarantee future compatibility from signatures alone.
Operator failures are not swallowed or retried after KV writes.

## Validation boundaries

FIA TND and BSND are **not bitwise equivalent**. The packed-V microtest measured
max absolute error 0.001953125 and relative RMS about 5.34e-5; near-tied logits
can change greedy tokens. Do not enable these options when exact baseline token
identity is required. Run task-specific quality evaluation: a small smoke/retrieval
corpus and sampled exact gating/GEMM tests do not prove model-level equivalence.
NZ validation includes in-place weight updates, ND storage and graph replay.

Use repeated same-device E2E tests across models/lengths, recording mean/median/p99
TTFT/TPOT and total/output throughput. Count unique implementation names in
`flaggems_ops.log`, frozen before extra quality/profiling requests. One-time
`FIA_TND_FAST_HIT`, `GATING24_FAST_HIT` and `PREFILL_NZ_FAST_HIT` messages confirm
dispatch, not operator counts; ranks and logging handlers may duplicate them.
