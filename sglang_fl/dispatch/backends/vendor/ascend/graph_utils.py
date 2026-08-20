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

"""Pure helpers for Ascend logical decode graph capture."""

from __future__ import annotations

import os

_EXACT_DECODE_GRAPH_ENV = "SGLANG_FL_EXACT_DECODE_GRAPH"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def exact_decode_graph_enabled() -> bool:
    """Return whether logical decode graph buckets are explicitly requested."""

    return os.getenv(_EXACT_DECODE_GRAPH_ENV, "").strip().lower() in _TRUE_VALUES


def align_capture_size(size: int, alignment: int) -> int:
    """Round a logical graph batch size up to its communication alignment."""

    if size <= 0 or alignment <= 0:
        raise ValueError(f"size and alignment must be positive: {size=}, {alignment=}")
    return ((size + alignment - 1) // alignment) * alignment


def resolve_logical_batch_size(
    original_batch_size: int | None, current_batch_size: int
) -> int:
    """Use the pre-padding size when SGLang provides one, including warmup None."""

    return (
        current_batch_size
        if original_batch_size is None
        else int(original_batch_size)
    )


def find_graph_metadata_backends(attn_backend: object) -> list[object]:
    """Return leaf attention backends that own per-batch graph metadata.

    Hybrid linear-attention backends are wrappers and keep their graph state in
    child backends.  Traverse ``attn_backend_list`` recursively while avoiding
    duplicate children; ordinary Ascend backends are returned directly.
    """

    result: list[object] = []
    visited: set[int] = set()

    def visit(backend: object) -> None:
        backend_id = id(backend)
        if backend_id in visited:
            return
        visited.add(backend_id)

        if hasattr(backend, "graph_metadata"):
            result.append(backend)
        for child in getattr(backend, "attn_backend_list", ()):
            visit(child)

    visit(attn_backend)
    return result
