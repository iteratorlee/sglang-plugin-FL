"""Moving independent inputs out of the recurrence preserves its FP32 state."""

import pytest
import torch

from test_glm5_kda_prefill import tensors


@pytest.mark.parametrize("lower_bound", [-1.0, -5.0, -12.0])
@pytest.mark.parametrize("length", [128, 2048])
def test_preparation_matches_head_parallel_path(monkeypatch, lower_bound, length):
    from sglang_fl.models.glm_53_flash.kda_recurrent_npu import (
        glm_kda_varlen_recurrent_npu as run,
    )

    args = tensors(length, 1.5)
    args["lower_bound"] = lower_bound
    args["scale"] = 0.17
    args["cu_seqlens"][-1] = length - 1  # Preserve the untouched padding row.
    state = torch.randn(6, 4, 128, 128, device="npu") * 0.2
    old, new = state.clone(), state.clone()
    monkeypatch.setenv("SGLANG_FL_GLM53_KDA_PREFILL_HEADS", "1")
    monkeypatch.setenv("SGLANG_FL_GLM53_KDA_PREFILL_PREPARE", "0")
    expected = run(**args, initial_state_source=old, prefill=True)
    monkeypatch.setenv("SGLANG_FL_GLM53_KDA_PREFILL_PREPARE", "1")
    actual = run(**args, initial_state_source=new, prefill=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, -1]) == 0


def test_preparation_keeps_reserved_zero_slot_unchanged(monkeypatch):
    from sglang_fl.models.glm_53_flash.kda_recurrent_npu import (
        glm_kda_varlen_recurrent_npu as run,
    )

    args = tensors(511)
    args["initial_state_indices"].zero_()
    state = torch.randn(6, 4, 128, 128, device="npu")
    old, new = state.clone(), state.clone()
    expected = run(**args, initial_state_source=old, prefill=False)
    monkeypatch.setenv("SGLANG_FL_GLM53_KDA_PREFILL_HEADS", "1")
    monkeypatch.setenv("SGLANG_FL_GLM53_KDA_PREFILL_PREPARE", "1")
    actual = run(**args, initial_state_source=new, prefill=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(new, state, rtol=0, atol=0)
