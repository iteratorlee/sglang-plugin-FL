#!/usr/bin/env python3
"""HYV4 shared-expert dense W8A8 graph replay checks."""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.utils import npu_format_cast


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(20260907)

    tokens, hidden, local_width = 32, 6144, 64
    static_x = torch.randn(
        tokens, hidden, device="npu", dtype=torch.bfloat16
    ) * 0.05
    # NPU dynamic W8A8 linear post-loading layout is [K,N].
    w1 = npu_format_cast(
        torch.randint(
            -3,
            4,
            (hidden, 2 * local_width),
            device="npu",
            dtype=torch.int8,
        )
    )
    s1 = (
        torch.rand(2 * local_width, device="npu", dtype=torch.float32)
        * 0.002
        + 0.0005
    ).to(torch.bfloat16)
    w2 = npu_format_cast(
        torch.randint(
            -3,
            4,
            (local_width, hidden),
            device="npu",
            dtype=torch.int8,
        )
    )
    s2 = (
        torch.rand(hidden, device="npu", dtype=torch.float32) * 0.002
        + 0.0005
    ).to(torch.bfloat16)
    def shared_expert(x):
        xq, xs = torch.ops.npu.npu_dynamic_quant(x)
        gate_up = torch.ops.npu.npu_quant_matmul(
            xq,
            w1,
            s1,
            pertoken_scale=xs.flatten(),
            output_dtype=x.dtype,
        )
        # This is SiluAndMul.forward_npu, the method selected by the model.
        activated = torch_npu.npu_swiglu(gate_up)
        aq, a_s = torch.ops.npu.npu_dynamic_quant(activated)
        return torch.ops.npu.npu_quant_matmul(
            aq,
            w2,
            s2,
            pertoken_scale=a_s.flatten(),
            output_dtype=x.dtype,
        )

    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_out = shared_expert(static_x)

    real_base = static_x[0].clone()
    steps = []
    last_replay = None
    last_input = None
    for step in range(8):
        changed = torch.zeros_like(static_x)
        real = real_base * (0.55 + 0.05 * step)
        changed[:2].copy_(real.unsqueeze(0).expand(2, -1))
        static_x.copy_(changed)
        graph.replay()
        torch_npu.npu.synchronize()
        replay = graph_out.clone()
        eager = shared_expert(changed)
        torch_npu.npu.synchronize()
        reference_error = (
            replay[:2].float() - eager[:2].float()
        ).abs().max().item()
        duplicate_error = (
            replay[0].float() - replay[1].float()
        ).abs().max().item()
        steps.append(
            {
                "step": step + 1,
                "graph_vs_eager_real_rows_max_abs_error": reference_error,
                "duplicate_real_rows_max_abs_error": duplicate_error,
                "passed": reference_error <= 0.01 and duplicate_error == 0.0,
            }
        )
        last_replay = replay
        last_input = changed

    # Change only the 30 padding rows. Their values must not affect either
    # identical real row through per-token quantization or dense GEMM.
    padding_changed = last_input.clone()
    padding_changed[2:].copy_(static_x.new_empty(tokens - 2, hidden).normal_() * 0.05)
    static_x.copy_(padding_changed)
    graph.replay()
    torch_npu.npu.synchronize()
    padding_replay = graph_out.clone()
    padding_invariance_error = (
        padding_replay[:2].float() - last_replay[:2].float()
    ).abs().max().item()
    padding_duplicate_error = (
        padding_replay[0].float() - padding_replay[1].float()
    ).abs().max().item()
    result = {
        "input_shape": list(static_x.shape),
        "gate_up_weight_shape": list(w1.shape),
        "down_weight_shape": list(w2.shape),
        "steps": steps,
        "padding_changed_real_rows_max_abs_error": padding_invariance_error,
        "padding_changed_duplicate_rows_max_abs_error": padding_duplicate_error,
        "passed": all(row["passed"] for row in steps)
        and padding_invariance_error <= 0.01
        and padding_duplicate_error == 0.0,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
