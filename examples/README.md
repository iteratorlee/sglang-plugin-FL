# Examples

Offline inference examples for smoke-testing sglang-plugin-FL with various models.

Each `*_offline_inference.py` script is self-contained: it loads the model, runs inference, validates output,
and exits with code 0 on success or non-zero on failure. This makes them directly
usable in CI pipelines.

## Available Examples

| Script | Model | Architecture | TP |
|--------|-------|-------------|-----|
| `qwen3_6_35b_a3b_offline_inference.py` | Qwen3.6-35B-A3B | MoE (256 experts) | 1 |
| `qwen3_6_27b_offline_inference.py` | Qwen3.6-27B | Dense (hybrid attention) | 1 |

## Usage

```bash
# Run with plugin loaded (default — SGLANG_PLUGINS auto-discovers sglang_fl)
python examples/qwen3_6_35b_a3b_offline_inference.py

# Run with only ATen replacement (no fused op dispatch)
SGLANG_FL_OOT_ENABLED=0 python examples/qwen3_6_35b_a3b_offline_inference.py

# Run baseline (no plugin at all)
SGLANG_PLUGINS=__none__ python examples/qwen3_6_35b_a3b_offline_inference.py

# Custom model path / TP
MODEL_PATH=/data/models/Qwen3.6-27B TP_SIZE=2 python examples/qwen3_6_27b_offline_inference.py
```

## CI Integration

The offline Python examples are designed for CI use:

- **Exit code**: 0 = pass, non-zero = fail
- **Model path**: Configurable via `MODEL_PATH` env var
- **Skip if model missing**: Prints message and exits with code 1
- **No interactive input**: Fully automated
- **Deterministic**: temperature=0, greedy decoding

A CI workflow can run them as:

```yaml
- name: Run inference examples
  run: |
    for f in examples/*_offline_inference.py; do
      echo "========== Running $f =========="
      python "$f" || exit 1
    done
```

## Test Modes

Control the plugin behavior via environment variables:

| Mode | Env Vars | What's tested |
|------|----------|---------------|
| Baseline | `SGLANG_PLUGINS=__none__` | Native SGLang, no plugin |
| ATen only | `SGLANG_FL_OOT_ENABLED=0` | Layer 1: FlagGems ATen replacement |
| Full | (default) | Layer 1 + Layer 2: ATen + fused op dispatch |
| Vendor | `USE_FLAGGEMS=0 SGLANG_FL_PREFER=vendor` | Layer 2 only: vendor fused ops |

For precision comparison across modes, use `tests/test_precision_align.py` or `tests/validate.sh`.


## MiniMax-M3.1 W8A8 on Ascend (client concurrency 64)

`minimax_m31_ascend_c64.sh` starts a persistent inference service using a TP16 / DP1 launch profile for the CANN 9.0 / TorchNPU 2.10 image with the MiniMax plugin patches installed. It increases server max running requests from 8 to 64 and captures full decode graphs for 1/2/4/8/16/24/32/48/64. Context length is 67584; chunked prefill is 4096; BF16 KV, plugin, FlagGems (`add`) and FlagTree stay enabled. Tested with image `m31-sglang-main-flagos:opt1-decode-merge` (based on `quay.io/ascend/sglang:main-cann9.0.0-a3`), SGLang `0.5.20.dev628+g7f1f8c706` and FlagTree `0.6.1+ascend3.5`. Prefix caching, MTP and overlap remain disabled. Keep the serial graph update, HCCL and indexer stability settings in the script.

Set `M31_MODEL_PATH`, `M31_PORT` or `M31_LOG_DIR` when needed, then run the script inside the prepared Ascend container. Server stdout/stderr should be redirected by the caller.

With the official SGLang benchmark at C64, 128 requests, exact 1024 output tokens and 8 warmups, this profile measured 1387.53 total tokens/s at 1024 input and 2370.48 total tokens/s at 4096 input. The previous max-running=8 configuration measured 388.40 and 804.06 respectively. Mean TPOT increases with the larger batch (1k: 40.34→86.92ms). Fourteen basic semantic cases passed. These are throughput measurements, not a full precision claim; long-input tuning is still in progress. This is a reproducible reference profile, not a claim of optimal settings for every workload.

### DP attention profile

`minimax_m31_ascend_dp_attention.sh` uses TP16 with DP attention. It defaults to DP4; set `M31_DP_SIZE=2` for DP2. Global max running requests remains 64. Local graph buckets are 4/8/12/16 for DP4 and 8/16/24/32 for DP2. The CLI chunk sizes 16384/8192 resolve to 4096 per attention group. The image, context, sampling and stability settings match the TP16/DP1 reference above. `M31_MODEL_PATH`, `M31_PORT` and `M31_LOG_DIR` have the same meaning.

With the official benchmark at C64, output1024 and 8 full-output warmups:

| Input | Formal requests | DP1 total tokens/s | DP2 | DP4 | DP4 total tokens/s/logical NPU |
|---|---:|---:|---:|---:|---:|
| 4096 | 128 | 2370.48 | 2890.36 | 3712.51 | 232.03 |
| 16384 | 64 (screen) | 3091.12 | 3750.04 | 5504.24 | 344.01 |
| 32768 | 64 (screen) | 3242.84 | 5433.20 | 6859.72 | 428.73 |

All profiles use 16 logical NPUs. DP4 passed 14 sequential and 64 concurrent basic semantic checks, and all 630 logged decode batches used graph. Long-input results are screening measurements; final validation with 128 requests remains pending. DP4 improves aggregate throughput, while mean TPOT at 32k increases from DP2's 133.34ms to 159.85ms. These checks do not establish full model precision equivalence.
