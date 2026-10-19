# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Copy the three BF16 head-128 caches without floating point arithmetic."""

import triton
import triton.language as tl


@triton.jit
def store_main_index_kv(k, v, index_k, slots, k_cache, v_cache, index_cache,
                        rows, capacity: tl.constexpr):
    dim = tl.arange(0, 128)
    for row in tl.range(tl.program_id(0), rows, tl.num_programs(0)):
        slot = tl.load(slots + row).to(tl.int64)
        # Slot zero is the native padded-token sink. Multiple dummy rows may
        # write it; its contents are deliberately unspecified, as in scatter.
        valid = (slot >= 0) & (slot < capacity)
        key = tl.load(k + row * 128 + dim)
        value = tl.load(v + row * 128 + dim)
        index = tl.load(index_k + row * 128 + dim)
        tl.store(k_cache + slot * 128 + dim, key, mask=valid)
        tl.store(v_cache + slot * 128 + dim, value, mask=valid)
        tl.store(index_cache + slot * 128 + dim, index, mask=valid)
