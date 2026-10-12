#!/usr/bin/env bash
set -eo pipefail
# Author: liyan <yan.l@pku.edu.cn>
# MiniMax-M3.1 TP16/DP4: 8192 prefill tokens per DP, full decode graph.
# Preserve the spare-slot profile's 17 request slots and 16 active requests.

export M31_CHUNKED_PREFILL_SIZE=${M31_CHUNKED_PREFILL_SIZE:-32768}
M31_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$M31_SCRIPT_DIR/minimax_m31_ascend_dp4_spare_slot.sh"
