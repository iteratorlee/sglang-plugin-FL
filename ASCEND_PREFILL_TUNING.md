# Ascend Qwen3.6 prefill tuning

These opt-in patches target the measured Qwen3.6-27B TP=2 BF16 prefill
shapes. They patch SGLang through the plugin; they do not modify SGLang,
FlagGems, or the installed `sgl_kernel_npu` package. All three options are
disabled by default. Enable them for **Qwen3.6-27B only**, **before starting a
new worker**:

```bash
export SGLANG_FL_FIA_TND_GQA=1
export SGLANG_FL_GDN_GATING_BLOCK24=1
export SGLANG_FL_PREFILL_NZ_GATE_UP=1
```

For Qwen3.6-35B-A3B, leave all three unset (or set them to `0`) and restart
the workers. Its shapes do not benefit, so there is no reason to install the
additional fallback wrappers. This does **not** disable the plugin, FlagGems,
or ordinary decode graphs. Keep the same per-model setting when benchmarking
or reproducing the selected configuration.

Keep the existing plugin/FlagGems configuration. These options do not enable
MTP, speculative decoding, prefix caching, or piecewise graphs. Validation
used ordinary decode graphs, TP=2, 64 concurrent requests, 1024 output tokens,
16384-token prefill chunks, and `mem_fraction_static=0.82`.

## Verified environment and selection

The measured environment is Ascend 910C, the
`quay.io/ascend/sglang:v0.5.11-cann8.5.0-a3` image, and the **installed**
`sgl_kernel_npu` 2026.3.1 package. An upstream source checkout is not a
substitute for checking the installed kernel ABI. Revalidate other CANN,
SGLang, kernel, and FlagGems versions before enabling these options.

| Option | Selected input | Implementation |
|---|---|---|
| FIA TND GQA | One fresh, unpadded 16384-token EXTEND; Q12/K2/V2, head dimension 256, BF16 | Call the existing FIA TND interface and return its output without the BSND output-buffer copy. Preserve the original KV write and causal mask/scale. |
| GDN gating BLOCK24 | Contiguous BF16 a/b `[16384,24]`, FP32 A_log, FP32/BF16 dt_bias | Reuse the installed gating kernel with all 24 heads in one group rather than three groups of eight. Preserve its FP32 outputs, beta and threshold. |
| Ephemeral NZ gate-up | Contiguous BF16 x `[16384,5120]`, ND weight `[17408,5120]`, no bias, exact unquantized method | Convert the current weight to a temporary FRACTAL_NZ tensor for this call, then run linear. Keep the original Parameter and ND storage unchanged. |

The FIA guard accepts the verified contiguous V and zero-copy packed V
stride `(7168,256,1)`. It rejects continuation/padded batches and unverified
attention modes. Gating and NZ also reject unverified shapes, dtypes and
strides. Qwen3.6-35B-A3B's TP=2 head/linear shapes do not select these fast
paths. Normal decode shapes use the existing implementations and graphs.

During graph capture the NZ wrapper calls the original ND linear implementation:
CANN's ND-to-NZ Identity conversion cannot be captured. This preserves the
ordinary decode graphs and allows an explicit prefill graph to use the stock
path. The measured eager prefill optimization remains enabled.

There is no persistent NZ weight cache: in-place weight updates are read by
the next call. `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1` prevents installation
of the NZ wrapper. Function/kernel signature checks prevent installation
against incompatible APIs. These checks do not prove compatibility with
every future implementation that happens to retain the same signature.
NPU operator failures are not swallowed or retried after a KV write.

## Numerical and performance validation

Changing FIA layout changes its internal BF16 tiling. TND and BSND are
**not bitwise equivalent**: the measured packed-V operator test had maximum
absolute difference 0.001953125 and relative RMS about 5.34e-5. Small changes
can flip greedy argmax near a tie, so token-for-token equivalence is not
promised. The combination also produced late output-token differences in
the extended smoke corpus. Run task-specific quality evaluation in addition
to operator tolerances; a small retrieval/smoke corpus is not a model-quality
benchmark. Do not enable the options when exact baseline token identity is
required.

The gating numerical tests cover BF16/FP32 bias and non-default beta/threshold.
NZ checks include ND Parameter/storage preservation, updated weights, and
graph replay. Passing sampled bitwise tests does not imply that every GEMM
input is bitwise identical across layouts.

Use repeated same-device end-to-end benchmarks, not single-operator speedups,
to decide whether to enable the options. Record mean/median/p99 TTFT and TPOT,
total/output throughput, graph use, completed requests, and actual FlagGems
implementation names observed in `flaggems_ops.log`. Freeze that inventory
before extra profiling/log-prob quality requests. Check the other context
lengths and models as well: a permanent NZ weight conversion was rejected
because its end-to-end decode performance regressed.

One-time `FIA_TND_FAST_HIT`, `GATING24_FAST_HIT`, and `PREFILL_NZ_FAST_HIT`
messages identify selected paths. Their line count is not an operator count:
multiple logging handlers and tensor-parallel ranks can repeat a message.

To roll back, unset the three options and restart the worker. No weight,
checkpoint, SGLang, or FlagGems source restoration is needed.
