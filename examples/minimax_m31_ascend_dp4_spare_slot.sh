#!/usr/bin/env bash
set -eo pipefail
# Author: liyan <yan.l@pku.edu.cn>
# MiniMax-M3.1 TP16/DP4: 17 request slots, 16 active requests per DP.
# Pinned SGLang main uses the PP micro-batch cap even with PP=1.

export M31_DP_SIZE=4
export M31_MAX_RUNNING_REQUESTS=68
export M31_PP_MAX_MICRO_BATCH_SIZE=16
# Keep fixed internal ports outside the host's 32768-60999 ephemeral range.
export M31_DIST_INIT_ADDR=${M31_DIST_INIT_ADDR:-127.0.0.1:29600}
export M31_NCCL_PORT=${M31_NCCL_PORT:-29620}
export SGLANG_FL_ASCEND_MINIMAX_ELIDE_SCORE_FILLS=1
export SGLANG_FL_ASCEND_MINIMAX_PREFILL_CAUSAL=1
export MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP=1

M31_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$M31_SCRIPT_DIR/minimax_m31_ascend_dp_attention.sh"
