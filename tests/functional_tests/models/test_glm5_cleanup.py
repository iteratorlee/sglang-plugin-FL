"""Focused regressions for behavior-preserving GLM cleanup changes."""

from __future__ import annotations

import ast
import inspect
import textwrap
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn


def _parsed_function(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _count_named_calls(function, name: str) -> int:
    tree = _parsed_function(function)
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


def test_glm5_cleanup_does_not_replace_global_deepseek_indexer():
    """The explicit GLM decoder replacement must not mutate other models."""

    import sglang.srt.models.deepseek_v2 as deepseek_v2
    from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
    from sglang_fl.models.register import patch_deepseek_dsa_compat

    original_init = deepseek_v2.DeepseekV2AttentionMLA.__init__
    for _ in range(2):
        patch_deepseek_dsa_compat()
        assert deepseek_v2.Indexer is Indexer
        assert deepseek_v2.DeepseekV2AttentionMLA.__init__ is original_init


def test_glm5_decoder_contains_one_explicit_kpool_construction():
    """Catch reintroduction of a second GLM-owned KPool construction path."""

    from sglang_fl.models.glm5_next import Glm5NextDecoderLayer

    assert _count_named_calls(Glm5NextDecoderLayer.__init__, "_build_glm_kpool_indexer") == 1


class _Communicator:
    def prepare_attn(self, hidden_states, residual, _forward_batch):
        return hidden_states, residual

    def maybe_prefetch_next_full_attention_kv(self, *_args):
        return None

    def prepare_mlp(self, hidden_states, residual, _forward_batch):
        return hidden_states, residual

    def should_fuse_mlp_allreduce_with_next_layer(self, _forward_batch):
        return False

    def should_use_reduce_scatter(self, _forward_batch):
        return False

    def postprocess_layer(self, hidden_states, residual, _forward_batch):
        return hidden_states, residual


class _Attention(nn.Module):
    def forward(self, *, hidden_states, **_kwargs):
        return hidden_states


class _CountingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, _forward_batch):
        self.calls += 1
        return hidden_states + 1


def test_glm5_decoder_calls_dense_and_moe_mlp_once(monkeypatch):
    """The merged dense/MoE call site must preserve one invocation per layer."""

    import sglang_fl.models.glm5_next as model_module

    class _CountingMoE(model_module.Glm5NextMoE):
        def __init__(self):
            nn.Module.__init__(self)
            self.calls = 0
            self.experts = SimpleNamespace(
                moe_runner_config=SimpleNamespace(inplace=True)
            )

        def forward(self, hidden_states, _forward_batch):
            self.calls += 1
            return hidden_states + 1

    monkeypatch.setattr(model_module, "clear_attn_inputs", lambda _context: None)
    monkeypatch.setattr(model_module, "get_attn_tp_context", lambda: object())
    monkeypatch.setattr(
        model_module,
        "get_forward",
        lambda: SimpleNamespace(scoped=lambda **_kwargs: nullcontext()),
    )

    forward_batch = SimpleNamespace()
    hidden_states = torch.zeros(2, 4)
    for mlp in (_CountingMLP(), _CountingMoE()):
        layer = object.__new__(model_module.Glm5NextDecoderLayer)
        nn.Module.__init__(layer)
        layer.is_linear_attn = True
        layer.dsa_enable_prefill_cp = False
        layer.mla_enable_prefill_cp = False
        layer.layer_scatter_modes = None
        layer.layer_communicator = _Communicator()
        layer.self_attn = _Attention()
        layer.mlp = mlp

        output, residual, topk = layer.forward(
            positions=torch.arange(2),
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=None,
        )

        assert mlp.calls == 1
        torch.testing.assert_close(output, hidden_states + 1)
        assert residual is None
        assert topk is None
