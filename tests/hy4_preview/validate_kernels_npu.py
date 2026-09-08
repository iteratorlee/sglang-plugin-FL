#!/usr/bin/env python3
"""HYV4 TP32-local projection correctness and NPUGraph replay checks."""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from sglang_fl.models.hy4_preview.hy4_triton_projection import (
    hy4_head_bmm,
    hy4_head_bmm_prefill,
)


def reference(a: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor | None):
    projected = torch.einsum("thk,hkn->thn", a.float(), weight.float())
    if gate is not None:
        projected = projected * gate.view_as(projected).float()
    return projected.to(torch.bfloat16)


def check_projection(
    name: str,
    tokens: int,
    in_width: int,
    out_width: int,
    gated: bool,
    input_tail: int = 0,
) -> dict:
    # TP32 rank-local MLA shapes:
    # q_nope @ w_kc: [T,2,192] x [2,192,512]
    # attention @ w_vc: [T,2,512] x [2,512,256], with elementwise gate.
    torch.manual_seed(20260904 + tokens + in_width + out_width)
    a_storage = torch.randn(
        tokens, 2, in_width + input_tail, dtype=torch.bfloat16, device="npu"
    )
    # q_nope is a 192-channel view split from [q_nope(192), q_rope(64)].
    # Preserve its physical 256-channel token/head strides in this test.
    a = a_storage[..., :in_width]
    if input_tail:
        # DeepSeek's absorbed w_kc keeps logical [H,K,N] after
        # [H,N,K].contiguous().transpose(1,2), i.e. stride [K*N,1,K].
        weight_storage = (
            torch.randn(
                2, out_width, in_width, dtype=torch.bfloat16, device="npu"
            )
            * 0.02
        )
        weight = weight_storage.transpose(1, 2)
    else:
        weight = (
            torch.randn(
                2, in_width, out_width, dtype=torch.bfloat16, device="npu"
            )
            * 0.02
        )
    gate = (
        torch.sigmoid(
            torch.randn(tokens, 2 * out_width, dtype=torch.bfloat16, device="npu")
        )
        if gated
        else None
    )
    # A duplicated pair catches token-row aliasing in partial BLOCK_M tiles;
    # a reference error alone can obscure the row-corruption pattern.
    duplicate_span = tokens // 2 if tokens in (10, 32) else 0
    if duplicate_span:
        a[duplicate_span : 2 * duplicate_span].copy_(a[:duplicate_span])
        if gate is not None:
            gate[duplicate_span : 2 * duplicate_span].copy_(gate[:duplicate_span])
    eager = hy4_head_bmm(a, weight, gate)
    prefill = hy4_head_bmm_prefill(a, weight, gate)
    expected = reference(a, weight, gate)
    torch_npu.npu.synchronize()
    eager_error = (eager.float() - expected.float()).abs().max().item()
    prefill_error = (prefill.float() - expected.float()).abs().max().item()
    prefill_finite = bool(torch.isfinite(prefill).all().item())
    eager_duplicate_error = (
        (
            eager[:duplicate_span].float()
            - eager[duplicate_span : 2 * duplicate_span].float()
        )
        .abs()
        .max()
        .item()
        if duplicate_span
        else 0.0
    )

    static_a_storage = torch.empty_like(a_storage)
    static_a = static_a_storage[..., :in_width]
    static_a.copy_(a)
    if input_tail:
        static_weight_storage = torch.empty_like(weight_storage)
        static_weight = static_weight_storage.transpose(1, 2)
        static_weight.copy_(weight)
    else:
        static_weight = weight.clone()
    static_gate = gate.clone() if gate is not None else None
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_output = hy4_head_bmm(static_a, static_weight, static_gate)

    # Replay with different data to prove all token/head/channel gate offsets
    # are read dynamically rather than baked into the capture.
    replay_a = a * 0.75
    replay_gate = (
        torch.sigmoid(gate.float() * 0.5).to(torch.bfloat16)
        if gate is not None
        else None
    )
    static_a.copy_(replay_a)
    if static_gate is not None:
        static_gate.copy_(replay_gate)
    graph.replay()
    torch_npu.npu.synchronize()
    replay = graph_output.clone()
    replay_expected = reference(replay_a, weight, replay_gate)
    replay_error = (replay.float() - replay_expected.float()).abs().max().item()
    replay_duplicate_error = (
        (
            replay[:duplicate_span].float()
            - replay[duplicate_span : 2 * duplicate_span].float()
        )
        .abs()
        .max()
        .item()
        if duplicate_span
        else 0.0
    )
    return {
        "projection": name,
        "tokens": tokens,
        "shape": list(eager.shape),
        "input_stride": list(a.stride()),
        "input_contiguous": a.is_contiguous(),
        "weight_stride": list(weight.stride()),
        "weight_contiguous": weight.is_contiguous(),
        "eager_max_abs_error": eager_error,
        "prefill_max_abs_error": prefill_error,
        "prefill_finite": prefill_finite,
        "eager_duplicate_max_abs_error": eager_duplicate_error,
        "graph_replay_max_abs_error": replay_error,
        "graph_replay_duplicate_max_abs_error": replay_duplicate_error,
        "passed": eager_error <= 0.125
        and prefill_error <= 0.125
        and prefill_finite
        and replay_error <= 0.125
        and eager_duplicate_error == 0.0
        and replay_duplicate_error == 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [
        check_projection(
            "q_nope_split_view_to_latent", tokens, 192, 512, False, input_tail=64
        )
        for tokens in (1, 5, 10, 31, 32, 33, 37, 1024)
    ]
    rows += [
        check_projection("latent_to_value_gated", tokens, 512, 256, True)
        for tokens in (1, 5, 10, 31, 32, 33, 37, 512, 513, 1024)
    ]
    result = {"passed": all(row["passed"] for row in rows), "cases": rows}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
