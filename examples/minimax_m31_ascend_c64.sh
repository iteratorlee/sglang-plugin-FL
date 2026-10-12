#!/usr/bin/env bash
set -eo pipefail
# Author: liyan <yan.l@pku.edu.cn>
# TP16 / DP1 reference profile for client concurrency 64.
# Keeps BF16 KV, prefix/MTP/overlap off, and the verified stability patches.

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTHONUNBUFFERED=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=180
export HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable
# TorchNPU 2.10 supports graph with task queue 0 or 1.
export TASK_QUEUE_ENABLE=0
export SGLANG_FL_ASCEND_STAGE_DIAGNOSTICS=1
export SGLANG_FL_ASCEND_SERIAL_GRAPH_UPDATE=1
export SGLANG_FL_ASCEND_TP_AIV=1
export SGLANG_FL_ASCEND_MINIMAX_VECTOR_INDEXER=1
export SGLANG_FL_ASCEND_MINIMAX_FUSED_MERGE=1
export SGLANG_FL_ASCEND_MINIMAX_PREFILL_META=0
export SYNC_TOKEN_IDS_ACROSS_TP=1
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_DIST_BACKEND=hccl
export SGLANG_FL_PREFER=vendor
export SGLANG_FL_DISAGG_FLAGCX=0
export USE_FLAGGEMS=1
export SGLANG_FL_FLAGOS_WHITELIST=add
export SGLANG_FLAGGEMS_RECORD=1
M31_LOG_DIR=${M31_LOG_DIR:-/data/m31-flagos-main/launch-c64}
mkdir -p "$M31_LOG_DIR"
export SGLANG_FLAGGEMS_LOG_PATH="$M31_LOG_DIR/gems_aten.log"

M31_MODEL_PATH=${M31_MODEL_PATH:-/public-flash/models/MiniMax-M3.1-preview-private-W8A8-FlagOS-20260929}
M31_PORT=${M31_PORT:-33013}

exec python3 -m sglang.launch_server \
  --model-path "$M31_MODEL_PATH" \
  --served-model-name MiniMax-M3.1 \
  --trust-remote-code \
  --device npu \
  --attention-backend ascend \
  --sampling-backend pytorch \
  --tp-size 16 \
  --nnodes 1 \
  --node-rank 0 \
  --random-seed 1012273410 --host 0.0.0.0 \
  --port "$M31_PORT" \
  --context-length 67584 \
  --page-size 128 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 64 \
  --max-running-requests 64 \
  --cuda-graph-bs-decode 1 2 4 8 16 24 32 48 64
