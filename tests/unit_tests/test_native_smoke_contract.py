"""CPU-only checks for the standalone NPU probe's oracle and topology input."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.manual import native_npu_collectives as smoke


def test_rank_oracle_uses_global_rank_and_destination():
    expected = smoke.expected_outputs(torch, [2, 5], 1, 3)
    left, _ = smoke.rank_inputs(torch, 2, 2, 3)
    right, _ = smoke.rank_inputs(torch, 5, 2, 3)
    torch.testing.assert_close(expected["group_all_reduce"], left + right)
    torch.testing.assert_close(
        expected["group_all_gather"], torch.cat([left, right], 0)
    )
    torch.testing.assert_close(expected["npu_all_gather"], torch.cat([left, right], 1))
    torch.testing.assert_close(expected["group_reduce_scatter"], left + right + 4)


def test_changed_replay_input_detects_stale_capture_result():
    group = SimpleNamespace(ranks=[2, 5], rank_in_group=1, unique_name="tp:0")
    outputs = smoke.expected_outputs(torch, group.ranks, group.rank_in_group, 1)
    fake = SimpleNamespace(
        npu=SimpleNamespace(synchronize=Mock()),
        testing=torch.testing,
        arange=torch.arange,
        float32=torch.float32,
        cat=torch.cat,
        stack=torch.stack,
    )
    smoke.assert_outputs(fake, group, outputs, 1)
    with pytest.raises(AssertionError):
        smoke.assert_outputs(fake, group, outputs, 2)
    assert fake.npu.synchronize.call_count == 2


@pytest.mark.parametrize(
    "groups",
    [
        {"tp": [[0, 1]]},
        {"tp": [[0, 1]], "attention_tp": [[0], [1]]},
        {"tp": [[0, 0]], "attention_tp": [[0, 1]]},
        {"tp": [[0, 2]], "attention_tp": [[0, 1]]},
    ],
)
def test_probe_refuses_implicit_or_invalid_topology(groups):
    with pytest.raises(ValueError):
        smoke.validate_groups(groups, 2)


def test_probe_preserves_explicit_group_partitions():
    groups = {"tp": [[0, 1, 2, 3]], "attention_tp": [[0, 1], [2, 3]]}
    assert smoke.validate_groups(groups, 4) is groups
