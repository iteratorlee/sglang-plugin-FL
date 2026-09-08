#!/usr/bin/env python3
"""HYV4 TP32-local W8A8 MoE eager/NPUGraph routing replay check."""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from sglang_fl.models.hy4_preview.hy4 import (
    _hy_npu_fused_experts_clamped,
    _hy_npu_fused_experts_w8a8_decode,
)


def run(fn, hidden, w13, s13, w2, s2, weights, ids):
    return fn(
        hidden_states=hidden,
        w13=w13,
        w13_scale=s13,
        w2=w2,
        w2_scale=s2,
        topk_weights=weights,
        topk_ids=ids,
        top_k=ids.shape[1],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(20260904)
    device = "npu"
    tokens, hidden_size, experts, topk, local_width = (
        args.tokens,
        6144,
        256,
        8,
        64,
    )
    hidden = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
    # Small integer values/scales avoid overflow while retaining the production
    # TP32 rank-local shapes and both A8 quantization points.
    w13 = torch.randint(
        -2, 3, (experts, 2 * local_width, hidden_size), dtype=torch.int8, device=device
    )
    s13 = torch.full(
        (experts, 2 * local_width), 0.001, dtype=torch.bfloat16, device=device
    )
    w2 = torch.randint(
        -2, 3, (experts, hidden_size, local_width), dtype=torch.int8, device=device
    )
    s2 = torch.full(
        (experts, hidden_size), 0.001, dtype=torch.bfloat16, device=device
    )
    # Only 32 of 256 experts are selected: this covers both empty experts and
    # multiple tokens routed to each active expert.
    ids = (
        torch.arange(tokens * topk, dtype=torch.int32, device=device)
        .view(tokens, topk)
        .remainder(32)
    )
    weights = torch.rand(tokens, topk, dtype=torch.bfloat16, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    eager = run(
        _hy_npu_fused_experts_clamped,
        hidden,
        w13,
        s13,
        w2,
        s2,
        weights,
        ids,
    )
    torch_npu.npu.synchronize()
    decode_eager = run(
        _hy_npu_fused_experts_w8a8_decode,
        hidden,
        w13,
        s13,
        w2,
        s2,
        weights,
        ids,
    )
    torch_npu.npu.synchronize()
    static_hidden = hidden.clone()
    static_weights = weights.clone()
    static_ids = ids.clone()
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_output = run(
            _hy_npu_fused_experts_w8a8_decode,
            static_hidden,
            w13,
            s13,
            w2,
            s2,
            static_weights,
            static_ids,
        )
    graph.replay()
    torch_npu.npu.synchronize()
    first_replay = graph_output.clone()
    first_error = (
        first_replay.float() - decode_eager.float()
    ).abs().max().item()

    changed_hidden = hidden * 0.5
    static_hidden.copy_(changed_hidden)
    graph.replay()
    torch_npu.npu.synchronize()
    changed_replay = graph_output.clone()
    changed_eager = run(
        _hy_npu_fused_experts_w8a8_decode,
        changed_hidden,
        w13,
        s13,
        w2,
        s2,
        weights,
        ids,
    )
    torch_npu.npu.synchronize()
    changed_error = (
        changed_replay.float() - changed_eager.float()
    ).abs().max().item()

    # Route every slot to the last expert and change all combine weights.  This
    # proves the captured graph consumes dynamic routing metadata instead of
    # replaying the expert distribution seen during capture.
    rerouted_ids = torch.full_like(ids, experts - 1)
    rerouted_weights = torch.linspace(
        1.0, 2.0, topk, dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    rerouted_weights = rerouted_weights.unsqueeze(0).expand(tokens, -1).contiguous()
    rerouted_weights = rerouted_weights / rerouted_weights.sum(dim=-1, keepdim=True)
    static_ids.copy_(rerouted_ids)
    static_weights.copy_(rerouted_weights)
    graph.replay()
    torch_npu.npu.synchronize()
    rerouted_replay = graph_output.clone()
    rerouted_eager = run(
        _hy_npu_fused_experts_w8a8_decode,
        changed_hidden,
        w13,
        s13,
        w2,
        s2,
        rerouted_weights,
        rerouted_ids,
    )
    torch_npu.npu.synchronize()
    rerouted_error = (
        rerouted_replay.float() - rerouted_eager.float()
    ).abs().max().item()

    # Decode graph pads every live batch to 32 rows.  Exercise the model-level
    # failure shape directly: rows 0/1 are identical real requests, rows 2..31
    # are padding, and routing/input metadata changes for eight replays.
    base_real_hidden = hidden[0].clone()
    bs2_steps = []
    last_bs2_replay = None
    last_bs2_hidden = None
    last_bs2_ids = None
    last_bs2_weights = None
    for step in range(8):
        bs2_hidden = torch.zeros_like(hidden)
        real_hidden = base_real_hidden * (0.55 + 0.05 * step)
        bs2_hidden[:2].copy_(real_hidden.unsqueeze(0).expand(2, -1))
        bs2_ids = torch.zeros_like(ids)
        real_ids = (
            torch.arange(topk, dtype=torch.int32, device=device) + 7 * step
        ).remainder(experts)
        bs2_ids[:2].copy_(real_ids.unsqueeze(0).expand(2, -1))
        if tokens > 2:
            padding_ids = (
                torch.arange(
                    (tokens - 2) * topk,
                    dtype=torch.int32,
                    device=device,
                ).view(tokens - 2, topk)
                + 13 * step
            ).remainder(experts)
            bs2_ids[2:].copy_(padding_ids)
        bs2_weights = torch.full_like(weights, 1.0 / topk)
        real_weights = torch.linspace(
            1.0, 2.0, topk, dtype=torch.float32, device=device
        ).to(torch.bfloat16)
        real_weights = real_weights / real_weights.sum()
        bs2_weights[:2].copy_(real_weights.unsqueeze(0).expand(2, -1))
        static_hidden.copy_(bs2_hidden)
        static_ids.copy_(bs2_ids)
        static_weights.copy_(bs2_weights)
        graph.replay()
        torch_npu.npu.synchronize()
        bs2_replay = graph_output.clone()
        bs2_eager = run(
            _hy_npu_fused_experts_w8a8_decode,
            bs2_hidden,
            w13,
            s13,
            w2,
            s2,
            bs2_weights,
            bs2_ids,
        )
        torch_npu.npu.synchronize()
        reference_error = (
            bs2_replay[:2].float() - bs2_eager[:2].float()
        ).abs().max().item()
        duplicate_error = (
            bs2_replay[0].float() - bs2_replay[1].float()
        ).abs().max().item()
        bs2_steps.append(
            {
                "step": step + 1,
                "graph_vs_eager_real_rows_max_abs_error": reference_error,
                "duplicate_real_rows_max_abs_error": duplicate_error,
                "passed": reference_error <= 0.01 and duplicate_error == 0.0,
            }
        )
        last_bs2_replay = bs2_replay
        last_bs2_hidden = bs2_hidden
        last_bs2_ids = bs2_ids
        last_bs2_weights = bs2_weights

    # Keep both real rows fixed while changing every padding row and its route.
    # The active output must be invariant to padding expert counts/order.
    padding_changed_hidden = last_bs2_hidden.clone()
    padding_changed_hidden[2:].copy_(hidden[2:] * 0.25)
    padding_changed_ids = last_bs2_ids.clone()
    padding_changed_ids[2:].fill_(experts - 1)
    padding_changed_weights = last_bs2_weights.clone()
    padding_changed_weights[2:].fill_(1.0 / topk)
    static_hidden.copy_(padding_changed_hidden)
    static_ids.copy_(padding_changed_ids)
    static_weights.copy_(padding_changed_weights)
    graph.replay()
    torch_npu.npu.synchronize()
    padding_changed_replay = graph_output.clone()
    padding_real_invariance_error = (
        padding_changed_replay[:2].float() - last_bs2_replay[:2].float()
    ).abs().max().item()
    padding_duplicate_error = (
        padding_changed_replay[0].float() - padding_changed_replay[1].float()
    ).abs().max().item()
    prefill_decode_error = (
        eager.float() - decode_eager.float()
    ).abs().max().item()
    result = {
        "shape": list(eager.shape),
        "active_experts": 32,
        "empty_experts": 224,
        "prefill_v1_vs_decode_v2_max_abs_error": prefill_decode_error,
        "capture_replay_max_abs_error": first_error,
        "changed_input_replay_max_abs_error": changed_error,
        "rerouted_expert": experts - 1,
        "rerouted_active_experts": 1,
        "rerouted_replay_max_abs_error": rerouted_error,
        "bs2_padding_8step_replays": bs2_steps,
        "padding_changed_real_rows_max_abs_error": padding_real_invariance_error,
        "padding_changed_duplicate_rows_max_abs_error": padding_duplicate_error,
        "passed": (
            prefill_decode_error <= 0.01
            and first_error <= 0.01
            and changed_error <= 0.01
            and rerouted_error <= 0.01
            and all(row["passed"] for row in bs2_steps)
            and padding_real_invariance_error <= 0.01
            and padding_duplicate_error == 0.0
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
