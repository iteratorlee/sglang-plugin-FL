# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend adapter for SGLang's out-of-tree DSA KV-pool factory contract."""

from typing import Optional

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool


class AscendDSATokenToKVPool(NPUMLATokenToKVPool):
    """Use the native NPU MLA pool for DSA models on an OOT platform.

    SGLang's OOT DSA builder passes ``kv_cache_dim`` because the CUDA/HIP DSA
    pool may use a packed cache layout.  Ascend's native DSA path deliberately
    reuses :class:`NPUMLATokenToKVPool`, which stores latent KV, RoPE KV and the
    DSA index key in separate NPU buffers and therefore does not consume that
    argument.  This adapter accepts the current OOT factory signature while
    preserving the native NPU layout and its ``set/get_index_k_buffer`` API.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        kv_cache_dim: int,
        index_head_dim: Optional[int],
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ) -> None:
        # ``kv_cache_dim`` is part of SGLang's generic OOT DSA contract.  The
        # native NPU pool uses its explicit latent/RoPE dimensions instead.
        del kv_cache_dim
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            index_head_dim=index_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )
