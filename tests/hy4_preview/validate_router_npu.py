#!/usr/bin/env python3
"""HYV4 FP32 router and grouped TopK graph replay checks."""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu


def route(hidden, weight, bias):
    logits = torch.nn.functional.linear(hidden.float(), weight)
    topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
        logits,
        k=8,
        bias=bias,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=0,
        norm_type=1,
        routed_scaling_factor=1,
        eps=float(1e-20),
    )
    return logits, topk_weights, topk_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(20260908)
    tokens, hidden_size, experts = 32, 6144, 256
    static_hidden = torch.randn(
        tokens, hidden_size, device="npu", dtype=torch.bfloat16
    ) * 0.05
    weight = torch.randn(
        experts, hidden_size, device="npu", dtype=torch.float32
    ) * 0.01
    bias = torch.randn(experts, device="npu", dtype=torch.float32) * 0.001
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_logits, graph_weights, graph_ids = route(static_hidden, weight, bias)

    real_base = static_hidden[0].clone()
    steps = []
    last_replay = None
    last_hidden = None
    for step in range(8):
        changed = torch.zeros_like(static_hidden)
        real = real_base * (0.55 + 0.05 * step)
        changed[:2].copy_(real.unsqueeze(0).expand(2, -1))
        static_hidden.copy_(changed)
        graph.replay()
        torch_npu.npu.synchronize()
        replay_logits = graph_logits.clone()
        replay_weights = graph_weights.clone()
        replay_ids = graph_ids.clone()
        eager_logits, eager_weights, eager_ids = route(changed, weight, bias)
        torch_npu.npu.synchronize()
        logits_reference_error = (
            replay_logits[:2] - eager_logits[:2]
        ).abs().max().item()
        weights_reference_error = (
            replay_weights[:2].float() - eager_weights[:2].float()
        ).abs().max().item()
        ids_reference_equal = bool(
            torch.equal(replay_ids[:2], eager_ids[:2])
        )
        logits_duplicate_error = (
            replay_logits[0] - replay_logits[1]
        ).abs().max().item()
        weights_duplicate_error = (
            replay_weights[0].float() - replay_weights[1].float()
        ).abs().max().item()
        ids_duplicate_equal = bool(torch.equal(replay_ids[0], replay_ids[1]))
        steps.append(
            {
                "step": step + 1,
                "logits_graph_vs_eager_max_abs_error": logits_reference_error,
                "weights_graph_vs_eager_max_abs_error": weights_reference_error,
                "ids_graph_vs_eager_equal": ids_reference_equal,
                "logits_duplicate_rows_max_abs_error": logits_duplicate_error,
                "weights_duplicate_rows_max_abs_error": weights_duplicate_error,
                "ids_duplicate_rows_equal": ids_duplicate_equal,
                "passed": logits_reference_error <= 0.01
                and weights_reference_error <= 0.01
                and ids_reference_equal
                and logits_duplicate_error == 0.0
                and weights_duplicate_error == 0.0
                and ids_duplicate_equal,
            }
        )
        last_replay = (replay_logits, replay_weights, replay_ids)
        last_hidden = changed

    padding_changed = last_hidden.clone()
    padding_changed[2:].copy_(
        torch.randn(
            tokens - 2, hidden_size, device="npu", dtype=torch.bfloat16
        )
        * 0.05
    )
    static_hidden.copy_(padding_changed)
    graph.replay()
    torch_npu.npu.synchronize()
    padding_logits = graph_logits.clone()
    padding_weights = graph_weights.clone()
    padding_ids = graph_ids.clone()
    padding_logits_error = (
        padding_logits[:2] - last_replay[0][:2]
    ).abs().max().item()
    padding_weights_error = (
        padding_weights[:2].float() - last_replay[1][:2].float()
    ).abs().max().item()
    padding_ids_equal = bool(torch.equal(padding_ids[:2], last_replay[2][:2]))
    result = {
        "hidden_shape": list(static_hidden.shape),
        "router_weight_shape": list(weight.shape),
        "steps": steps,
        "padding_changed_logits_max_abs_error": padding_logits_error,
        "padding_changed_weights_max_abs_error": padding_weights_error,
        "padding_changed_ids_equal": padding_ids_equal,
        "passed": all(row["passed"] for row in steps)
        and padding_logits_error == 0.0
        and padding_weights_error == 0.0
        and padding_ids_equal,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
