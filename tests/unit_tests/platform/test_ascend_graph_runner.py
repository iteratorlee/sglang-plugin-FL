# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for Ascend logical decode graph bucket support."""

from __future__ import annotations

from types import SimpleNamespace


def test_exact_decode_graph_env_is_opt_in(monkeypatch) -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.graph_utils import (
        exact_decode_graph_enabled,
    )

    monkeypatch.delenv("SGLANG_FL_EXACT_DECODE_GRAPH", raising=False)
    assert exact_decode_graph_enabled() is False

    monkeypatch.setenv("SGLANG_FL_EXACT_DECODE_GRAPH", "YeS")
    assert exact_decode_graph_enabled() is True


def test_sub_tp_capture_size_is_communication_aligned() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.graph_utils import (
        align_capture_size,
    )

    assert [align_capture_size(size, 16) for size in [1, 2, 4, 8, 16, 32, 64]] == [
        16,
        16,
        16,
        16,
        16,
        32,
        64,
    ]


def test_logical_batch_size_falls_back_for_warmup_none() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.graph_utils import (
        resolve_logical_batch_size,
    )

    assert resolve_logical_batch_size(None, 1) == 1
    assert resolve_logical_batch_size(4, 16) == 4


def test_graph_metadata_backends_traverse_hybrid_children_once() -> None:
    from sglang_fl.dispatch.backends.vendor.ascend.graph_utils import (
        find_graph_metadata_backends,
    )

    full_attention = SimpleNamespace(graph_metadata={2: object()})
    linear_attention = SimpleNamespace(state_indices_list=[])
    nested = SimpleNamespace(
        attn_backend_list=[full_attention, linear_attention, full_attention]
    )
    hybrid = SimpleNamespace(attn_backend_list=[nested])

    assert find_graph_metadata_backends(hybrid) == [full_attention]
