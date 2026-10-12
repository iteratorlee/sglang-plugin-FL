#!/usr/bin/env bash
set -eo pipefail
# Author: liyan <yan.l@pku.edu.cn>
# TP16 / DP attention 2 or 4, global batch64 / full decode graph.
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
M31_LOG_DIR=${M31_LOG_DIR:-/data/m31-flagos-main/launch-dp-attention}
mkdir -p "$M31_LOG_DIR"
export SGLANG_FLAGGEMS_LOG_PATH="$M31_LOG_DIR/gems_aten.log"

M31_MODEL_PATH=${M31_MODEL_PATH:-/public-flash/models/MiniMax-M3.1-preview-private-W8A8-FlagOS-20260929}
M31_PORT=${M31_PORT:-33013}
M31_DP_SIZE=${M31_DP_SIZE:-4}
M31_MAX_RUNNING_REQUESTS=${M31_MAX_RUNNING_REQUESTS:-64}
M31_PP_ARGS=()
if [[ -n "${M31_PP_MAX_MICRO_BATCH_SIZE:-}" ]]; then
  M31_PP_ARGS=(--pp-max-micro-batch-size "$M31_PP_MAX_MICRO_BATCH_SIZE")
fi
M31_NETWORK_ARGS=()
if [[ -n "${M31_DIST_INIT_ADDR:-}" ]]; then
  M31_NETWORK_ARGS+=(--dist-init-addr "$M31_DIST_INIT_ADDR")
fi
if [[ -n "${M31_NCCL_PORT:-}" ]]; then
  M31_NETWORK_ARGS+=(--nccl-port "$M31_NCCL_PORT")
fi
case "$M31_DP_SIZE" in
  2) M31_GRAPH_BUCKETS=(8 16 24 32); M31_GRAPH_MAX=32; M31_CHUNK_SIZE=8192 ;;
  4) M31_GRAPH_BUCKETS=(4 8 12 16); M31_GRAPH_MAX=16; M31_CHUNK_SIZE=16384 ;;
  *) echo "M31_DP_SIZE must be 2 or 4" >&2; exit 2 ;;
esac

exec python3 -m sglang.launch_server \
  --model-path "$M31_MODEL_PATH" \
  --served-model-name MiniMax-M3.1 \
  --trust-remote-code \
  --device npu \
  --attention-backend ascend \
  --sampling-backend pytorch \
  --tp-size 16 \
  --dp-size "$M31_DP_SIZE" \
  --enable-dp-attention \
  --nnodes 1 \
  --node-rank 0 \
  "${M31_NETWORK_ARGS[@]}" \
  --random-seed 1012273410 --host 0.0.0.0 \
  --port "$M31_PORT" \
  --context-length 67584 \
  --page-size 128 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size "$M31_CHUNK_SIZE" \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode "$M31_GRAPH_MAX" \
  --max-running-requests "$M31_MAX_RUNNING_REQUESTS" \
  "${M31_PP_ARGS[@]}" \
  --cuda-graph-bs-decode "${M31_GRAPH_BUCKETS[@]}"
