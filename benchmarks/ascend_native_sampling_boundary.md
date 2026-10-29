# Native Ascend speculative-sampling boundary repair

The audited `sgl-kernel-npu==2026.9.0` target-only sampler accepted a draft with
`coin <= probability / acceptance_threshold`. Uniform random values have a
half-open `[0, 1)` range and may equal zero. A zero-probability draft could
therefore be accepted. Use strict `<`; retain the explicit single-token
threshold shortcut and every other calculation.

`sglang_fl.native_sampling.configure()` installs the correction during native
plugin initialization, before allocator warmup and graph capture. It verifies
the exact installed source SHA256, JIT source, public entry point and function
ABI. It then builds a new JIT using public `triton.jit`, preserving module
globals and public entry-point aliases. Installation is atomic and idempotent;
unsupported revisions fail closed. No installed SGLang/kernel source is
rewritten and no per-token wrapper or additional kernel launch is introduced.

## Validation

```bash
python -m pytest -q tests/unit_tests/test_native_sampling.py
```

The CPU tests cover the zero-probability boundary, other threshold semantics,
exactly one changed AST comparison, ABI/source guards, initialization order,
concurrent installation, failure atomicity and real JIT/cache-key construction
when Triton is available. The source fixture is the exact audited kernel file;
its SHA256 is asserted by the tests.

Real Ascend 910B / PyTorch-NPU 2.10 / FlagTree 0.7.0+ascend3.5 / Triton-module
3.5.1 auxiliary-device verification on 2026-10-29:

| Check | Result |
|---|---|
| Vocabulary 16, 65,536 samples | 7 checks passed, including zero-probability boundary |
| Vocabulary 154,880, batch 14, 8,204 samples | 8 checks passed, including top-p and graph replay |
| Different conditional distributions per request/step, batches 1/2/14 | 99/99 CPU-reference paths matched across 16 calls, with graph probability staging |
| Native RNG / CPU coins, target-only and chain sampling | Distribution and exact supplied-coin paths passed |

Paired device-event timings include the sampler, output-buffer resets and
submission gaps; they are not isolated-kernel or full-model timings. Eight
alternating original/fixed pairs of 64 calls were measured after warmup.

| Batch | Original median ms/call | Fixed median ms/call |
|---:|---:|---:|
| 1 | 0.439231 | 0.439228 |
| 14 | 0.442984 | 0.440686 |

Non-boundary outputs were identical. No added overhead was observed in this
bounded test. The defect also exists under the older compiler: these checks
do **not** establish that it caused a difference in GPQA accuracy, and do not
replace complete model-level evaluation. Production services were unchanged
during this validation.
