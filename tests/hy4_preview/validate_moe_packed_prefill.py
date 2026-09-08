#!/usr/bin/env python3
"""Check HYV4 prefill MoE row independence with global tail padding."""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from sglang_fl.models.hy4_preview.hy4 import _hy_npu_fused_experts_clamped


def run(hidden, w13, s13, w2, s2, weights, ids):
    return _hy_npu_fused_experts_clamped(
        hidden_states=hidden,
        w13=w13,
        w13_scale=s13,
        w2=w2,
        w2_scale=s2,
        topk_weights=weights,
        topk_ids=ids,
        top_k=ids.shape[1],
    )


def max_error(left, right):
    return (left.float() - right.float()).abs().max().item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(20260904)
    device = "npu"
    hidden_size, experts, topk, local_width = 6144, 256, 8, 64
    w13 = torch.randint(
        -2,
        3,
        (experts, 2 * local_width, hidden_size),
        dtype=torch.int8,
        device=device,
    )
    s13 = torch.full(
        (experts, 2 * local_width),
        0.001,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.randint(
        -2,
        3,
        (experts, hidden_size, local_width),
        dtype=torch.int8,
        device=device,
    )
    s2 = torch.full(
        (experts, hidden_size), 0.001, dtype=torch.bfloat16, device=device
    )

    base_hidden = (
        torch.randn(5, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
    )
    base_ids = (
        torch.arange(5 * topk, dtype=torch.int32, device=device)
        .view(5, topk)
        .remainder(32)
    )
    base_weights = torch.rand(5, topk, dtype=torch.bfloat16, device=device)
    base_weights = base_weights / base_weights.sum(dim=-1, keepdim=True)
    hidden10 = torch.cat((base_hidden, base_hidden), dim=0)
    ids10 = torch.cat((base_ids, base_ids), dim=0)
    weights10 = torch.cat((base_weights, base_weights), dim=0)
    hidden128 = torch.cat(
        (
            hidden10,
            torch.zeros(118, hidden_size, dtype=torch.bfloat16, device=device),
        ),
        dim=0,
    )
    ids128 = torch.cat(
        (
            ids10,
            torch.zeros(118, topk, dtype=torch.int32, device=device),
        ),
        dim=0,
    )
    padding_weights = torch.full(
        (118, topk), 1.0 / topk, dtype=torch.bfloat16, device=device
    )
    weights128 = torch.cat((weights10, padding_weights), dim=0)

    out5 = run(base_hidden, w13, s13, w2, s2, base_weights, base_ids)
    out10 = run(hidden10, w13, s13, w2, s2, weights10, ids10)
    out128 = run(hidden128, w13, s13, w2, s2, weights128, ids128)
    torch_npu.npu.synchronize()
    result = {
        "shape": {
            "independent": list(out5.shape),
            "packed_real": list(out10.shape),
            "packed_with_tail_padding": list(out128.shape),
        },
        "identical_sequence_rows_max_abs_error": max_error(out10[:5], out10[5:10]),
        "packed_vs_independent_max_abs_error": max_error(
            out10, torch.cat((out5, out5), dim=0)
        ),
        "tail_padding_real_rows_max_abs_error": max_error(out128[:10], out10),
    }
    result["passed"] = all(
        value <= 0.01 for key, value in result.items() if key.endswith("error")
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
