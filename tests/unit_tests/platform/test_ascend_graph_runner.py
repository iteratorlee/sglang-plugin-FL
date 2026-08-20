# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for Ascend logical decode graph bucket support."""

from __future__ import annotations


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
