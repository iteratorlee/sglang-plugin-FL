#!/usr/bin/env bash
set -eo pipefail
# Author: liyan <yan.l@pku.edu.cn>

# MiniMax-M3.1 W8A8, 16 logical NPUs: attention TP4 x DP4, routed EP16.
# Requires prepared image m31-sglang-main-flagos:kv-audit-v2, immutable ID
# sha256:219459534d6a8e2bf728f056b6cd58f3d5612e02aadaf5ca045fff68b5d8f772.
# This image contains selected plugin files; the whole Git HEAD is not validated.
# The R6 deployment profile passed complete functional/official benchmark gates.
# This portable wrapper is shell-checked; the exact deployed launcher is archived
# with the workspace report. Import isolation/preflight here differ from R6.
# Caller owns stdout/stderr redirection to a fresh run directory.
M31_MODEL_PATH=${M31_MODEL_PATH:-/public-flash/models/MiniMax-M3.1-preview-private-W8A8-FlagOS-20260929}
M31_PORT=${M31_PORT:-33013}
M31_LOG_DIR=${M31_LOG_DIR:-/data/m31-flagos-main/ep16-$(date +%Y%m%d-%H%M%S)-$$}
M31_DIST_INIT_ADDR=${M31_DIST_INIT_ADDR:-127.0.0.1:29600}
M31_NCCL_PORT=${M31_NCCL_PORT:-29620}
export M31_DEEPEP_API_DIR=${M31_DEEPEP_API_DIR:-/usr/local/python3.11.15/lib/python3.11/site-packages/deep_ep/vendors/hwcomputing/op_api/lib}
for value in "$M31_MODEL_PATH" "$M31_LOG_DIR" "$M31_DEEPEP_API_DIR"; do
  if [[ "$value" != /* ]]; then
    echo "Model, log and custom API paths must be absolute: $value" >&2
    exit 1
  fi
done
mkdir -p "$M31_LOG_DIR"
if [[ -e "$M31_LOG_DIR/audit" || -e "$M31_LOG_DIR/gems_aten.log" || -e "$M31_LOG_DIR/custom-api-preflight.json" ]]; then
  echo "Choose a fresh M31_LOG_DIR; run artifacts already exist." >&2
  exit 1
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Keep the prepared image imports independent of the caller's source checkout.
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/cann-9.0.0/python/site-packages:/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe
export PYTHONNOUSERSITE=1
unset PYTHONOPTIMIZE PYTHONHOME
cd /
export PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=180
export HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable
# TorchNPU 2.10 supports graph with task queue 0 or 1.
export TASK_QUEUE_ENABLE=0
export SGLANG_FL_ASCEND_STAGE_DIAGNOSTICS=1
export SGLANG_FL_ASCEND_SERIAL_GRAPH_UPDATE=1
export SGLANG_FL_ASCEND_SCHEDULER_CPU_PARTITION=0
export SGLANG_FL_ASCEND_MINIMAX_DECODE_PREFETCH=0
export DEEP_USE_MODE=default
export LD_LIBRARY_PATH="$M31_DEEPEP_API_DIR:$LD_LIBRARY_PATH"
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=768
export DEEPEP_HCCL_BUFFSIZE=768
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_LONG_SEQ_ROUND=1
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=0
export SGLANG_FL_ASCEND_MINIMAX_FUSED_KV_STORE=1
export SGLANG_FL_ASCEND_GRAPH_AUDIT=1
export SGLANG_FL_ASCEND_GRAPH_AUDIT_DIR="$M31_LOG_DIR/audit"
export SGLANG_FL_ASCEND_MINIMAX_EMPTY_GRAPH_UPDATE=0
export SGLANG_FL_ASCEND_TP_AIV=1
export SGLANG_FL_ASCEND_MINIMAX_VECTOR_INDEXER=1
export SGLANG_FL_ASCEND_MINIMAX_FUSED_MERGE=1
export SGLANG_FL_ASCEND_MINIMAX_ELIDE_SCORE_FILLS=1
export SGLANG_FL_ASCEND_MINIMAX_PREFILL_META=0
export SGLANG_FL_ASCEND_MINIMAX_PREFILL_CAUSAL=1
export SGLANG_FL_ASCEND_MINIMAX_PREFILL_FUTURE_SKIP=1
export MINIMAX_NPU_PREFILL_SCORE_BLOCKS_PER_STEP=1
export SYNC_TOKEN_IDS_ACROSS_TP=1
export SGLANG_PLUGINS=sglang_fl
export SGLANG_FL_DIST_BACKEND=hccl
export SGLANG_FL_PREFER=vendor
export SGLANG_FL_DISAGG_FLAGCX=0
export USE_FLAGGEMS=1
export SGLANG_FL_FLAGOS_WHITELIST=add
export SGLANG_FLAGGEMS_RECORD=1
export SGLANG_FLAGGEMS_LOG_PATH="$M31_LOG_DIR/gems_aten.log"

# Check ELF lookup before importing Torch or starting model workers.
python3 -S - "$M31_LOG_DIR/custom-api-preflight.json" <<'PY_API'
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

path = Path(os.environ['M31_DEEPEP_API_DIR']).resolve()
library_path = path / 'libcust_opapi.so'
if not library_path.is_file() or (path / 'libopapi.so').exists():
    raise RuntimeError('Expected the prepared custom API library without a base-library override')
expected_sha = 'f1e57f1caf5c7a8116de3fdcdc73a599cf0e36543d7d9e06351d18538854aee3'
sha = hashlib.sha256(library_path.read_bytes()).hexdigest()
if sha != expected_sha:
    raise RuntimeError(f'Unexpected custom API SHA: {sha}')
library = ctypes.CDLL('libcust_opapi.so', mode=ctypes.RTLD_LOCAL)
symbols = ['aclnn' + operation + suffix
           for operation in ('MoeLowLatencyDispatchV2', 'MoeLowLatencyCombineV2',
                             'CamMoeDispatchNormal', 'CamMoeCombineNormal')
           for suffix in ('', 'GetWorkspaceSize')]
for symbol in symbols:
    getattr(library, symbol)
loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
          if 'libcust_opapi.so' in line}
if loaded != {str(library_path)}:
    raise RuntimeError(f'Unexpected loaded custom API paths: {loaded}')
report = {'at': datetime.datetime.now().astimezone().isoformat(), 'passed': True,
          'library': str(library_path), 'sha256': sha, 'loaded_paths': sorted(loaded),
          'symbols': symbols, 'cwd': os.getcwd(), 'pythonpath': os.environ['PYTHONPATH'],
          'scope': 'ELF lookup only; does not validate operator math or whole-model correctness'}
with Path(sys.argv[1]).open('x') as output:
    output.write(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
PY_API

exec python3 -m sglang.launch_server \
  --model-path "$M31_MODEL_PATH" \
  --served-model-name MiniMax-M3.1 \
  --trust-remote-code \
  --device npu \
  --attention-backend ascend \
  --sampling-backend pytorch \
  --tp-size 16 \
  --dp-size 4 \
  --ep-size 16 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --deepep-dispatcher-output-dtype int8 \
  --enable-dp-attention \
  --nnodes 1 \
  --node-rank 0 \
  --dist-init-addr "$M31_DIST_INIT_ADDR" \
  --nccl-port "$M31_NCCL_PORT" \
  --random-seed 1012273410 --host 0.0.0.0 \
  --port "$M31_PORT" \
  --context-length 67584 \
  --page-size 128 \
  --mem-fraction-static 0.92 \
  --max-total-tokens 477184 \
  --chunked-prefill-size 32768 \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 16 \
  --max-running-requests 68 \
  --pp-max-micro-batch-size 16 \
  --cuda-graph-bs-decode 4 8 12 16
