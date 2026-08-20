from types import SimpleNamespace

import pytest
import torch

from sglang_fl.dispatch.backends.flagos.impl.topk import topk_flagos
from sglang_fl.dispatch.backends.vendor.ascend.impl.fused_moe import fused_moe_ascend


def test_flagos_topk_supports_current_top_k_field(monkeypatch) -> None:
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 4)
    obj = SimpleNamespace(
        topk_config=SimpleNamespace(
            top_k=2,
            renormalize=True,
            use_grouped_topk=False,
            correction_bias=None,
            scoring_func="softmax",
        )
    )

    def fake_topk_softmax(weights, ids, token_indices, logits, renormalize):
        assert logits is router_logits
        assert renormalize is True
        weights.fill_(0.5)
        ids.copy_(torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32))
        token_indices.zero_()

    monkeypatch.setattr("flag_gems.topk_softmax", fake_topk_softmax)

    result = topk_flagos(obj, hidden_states, router_logits)

    assert result.topk_weights.shape == (3, 2)
    assert result.topk_ids.shape == (3, 2)
    assert result.router_logits is router_logits


def test_flagos_topk_rejects_routing_modes_it_cannot_represent() -> None:
    obj = SimpleNamespace(
        topk_config=SimpleNamespace(
            top_k=2,
            renormalize=True,
            use_grouped_topk=True,
            correction_bias=None,
            scoring_func="softmax",
        )
    )

    with pytest.raises(NotImplementedError, match="plain softmax routing"):
        topk_flagos(obj, torch.randn(2, 4), torch.randn(2, 8))


def test_ascend_fused_moe_delegates_current_structured_dispatch_output() -> None:
    expected = object()
    layer = object()
    dispatch_output = SimpleNamespace(group_list_type=1)

    class FakeRunner:
        def __init__(self):
            self.calls = []

        def run(self, *args):
            self.calls.append(args)
            return expected

    runner = FakeRunner()
    obj = SimpleNamespace(runner=runner)

    assert fused_moe_ascend(obj, layer, dispatch_output) is expected
    assert runner.calls == [(dispatch_output, layer)]
