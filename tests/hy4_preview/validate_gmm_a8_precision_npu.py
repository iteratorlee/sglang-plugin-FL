"""Compare late FP32 A8 scaling with the former BF16 reconstruction path.

Stand-alone TP32-local expert component check; this does not change a service.
Checks both production GEMM shapes and replays with changed A8/scales/routing.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu


def reference(xq, w, ws, ts, counts):
    # CPU FP32 accumulation provides an independent reference for all 8 rows.
    xq, ws, ts = xq.cpu().float(), ws.cpu().float(), ts.cpu().float()
    result = []
    offset = 0
    for expert, count in enumerate(counts.cpu().tolist()):
        if count:
            accum = xq[offset:offset + count] @ w[expert].cpu().float().T
            result.append(accum * ts[offset:offset + count, None] * ws[expert])
            offset += count
    return torch.cat(result)


def compare(actual, expected):
    actual = actual.cpu().float()
    err = (actual - expected).abs()
    return {
        "max_abs_error": err.max().item(),
        "mean_abs_error": err.mean().item(),
        "normalized_max_error": err.max().item() / expected.abs().max().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def run_stage(module, k_dim, n_dim):
    tokens, experts = 8, 256
    w = torch.randint(-127, 128, (experts, n_dim, k_dim), dtype=torch.int8, device="npu")
    ws = (torch.rand(experts, n_dim, device="npu") * .004 + .0001).bfloat16()
    x = torch.randn(tokens, k_dim, device="npu", dtype=torch.bfloat16)
    xq, ts = torch.ops.npu.npu_dynamic_quant(x)
    counts = torch.zeros(experts, dtype=torch.int64, device="npu")
    counts[:tokens] = 1

    def call(q, scales, groups):
        return module.hy4_expert_gmm_i8(q, w, ws, groups, pertoken_scale=scales)

    for _ in range(2):
        call(xq, ts, counts)
    torch.npu.synchronize()
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_output = call(xq, ts, counts)
    records = []
    for step in range(4):
        if step:
            source = torch.randn_like(x) * (0.13 + step)
            changed_q, changed_ts = torch.ops.npu.npu_dynamic_quant(source)
            xq.copy_(changed_q)
            ts.copy_(changed_ts)
            counts.zero_()
            if step == 2:
                counts[-1] = tokens
            else:
                counts[step * 8:step * 8 + tokens] = 1
        graph.replay()
        torch.npu.synchronize()
        captured = graph_output.clone()
        eager = call(xq, ts, counts)
        reconstructed = xq.bfloat16() * ts.bfloat16().unsqueeze(1)
        previous = module.hy4_expert_gmm_i8(reconstructed, w, ws, counts)
        native = torch.ops.npu.npu_grouped_matmul(
            x=[xq], weight=[w.transpose(1, 2)], scale=[ws],
            per_token_scale=[ts], group_list=counts, split_item=2,
            group_type=0, group_list_type=1, output_dtype=torch.bfloat16,
        )[0]
        torch.npu.synchronize()
        expected = reference(xq, w, ws, ts, counts)
        record = {
            "step": step,
            "late_fp32_scaling": compare(eager, expected),
            "previous_bf16_reconstruction": compare(previous, expected),
            "native_gmm": compare(native, expected),
            "graph_eager_max_error": (captured.float() - eager.float()).abs().max().item(),
            "native_agreement_fraction": (eager == native).float().mean().item(),
        }
        record["passed"] = (
            record["late_fp32_scaling"]["finite"]
            and record["late_fp32_scaling"]["normalized_max_error"] < .004
            and record["graph_eager_max_error"] == 0
            and record["late_fp32_scaling"]["mean_abs_error"]
            <= record["previous_bf16_reconstruction"]["mean_abs_error"]
        )
        records.append(record)
        print(json.dumps({"shape": [experts, n_dim, k_dim], **record}), flush=True)
    return {"shape": [experts, n_dim, k_dim], "records": records}


def run_routed_chain(module, projection):
    tokens, experts, topk = 1, 256, 8
    w13 = torch.randint(-127, 128, (experts, 128, 6144), dtype=torch.int8, device="npu")
    w2 = torch.randint(-127, 128, (experts, 6144, 64), dtype=torch.int8, device="npu")
    s13 = (torch.rand(experts, 128, device="npu") * .004 + .0001).bfloat16()
    s2 = (torch.rand(experts, 6144, device="npu") * .004 + .0001).bfloat16()
    hidden = torch.randn(tokens, 6144, dtype=torch.bfloat16, device="npu")
    ids = torch.arange(topk, dtype=torch.int32, device="npu").view(tokens, topk)
    probs = torch.softmax(torch.rand(tokens, topk, device="npu"), -1).bfloat16()

    def chain(mode):
        q, row_indices, counts, ts = torch.ops.npu.npu_moe_init_routing_v2(
            hidden, ids, active_num=tokens * topk, expert_num=experts,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[0, experts], quant_mode=1,
        )
        counts = counts.to(torch.int64)

        def gmm(q, w, ws, scales):
            if mode == "native":
                return torch.ops.npu.npu_grouped_matmul(
                    x=[q], weight=[w.transpose(1, 2)], scale=[ws],
                    per_token_scale=[scales], group_list=counts,
                    split_item=2, group_type=0, group_list_type=1,
                    output_dtype=torch.bfloat16,
                )[0]
            if mode == "previous":
                x = q.bfloat16() * scales.bfloat16().unsqueeze(1)
                return module.hy4_expert_gmm_i8(x, w, ws, counts)
            return module.hy4_expert_gmm_i8(q, w, ws, counts, pertoken_scale=scales)

        g1 = gmm(q, w13, s13, ts)
        activation = projection.hy4_clamped_swiglu(g1)
        aq, activation_scale = torch.ops.npu.npu_dynamic_quant(activation)
        g2 = gmm(aq, w2, s2, activation_scale)
        result = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=g2, sorted_indices=row_indices.abs(), probs=probs,
        )
        return g1, activation, g2, result

    for _ in range(2):
        chain("late")
    torch.npu.synchronize()
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_outputs = chain("late")
    records = []
    for step in range(4):
        if step:
            hidden.copy_(torch.randn_like(hidden) * (step + .13))
            ids.copy_((torch.arange(topk, device="npu") + step * 61).int().view_as(ids))
            probs.copy_(torch.softmax(torch.randn_like(probs).float(), -1).bfloat16())
        graph.replay()
        torch.npu.synchronize()
        captured = [tensor.clone() for tensor in graph_outputs]
        eager = chain("late")
        native = chain("native")
        previous = chain("previous")
        torch.npu.synchronize()
        stages = {}
        for name, graph_tensor, actual, reference_tensor, old in zip(
            ("gmm1", "activation", "gmm2", "final"), captured, eager, native, previous
        ):
            stages[name] = {
                "graph_eager_max_error": (graph_tensor.float() - actual.float()).abs().max().item(),
                "native_max_error": (reference_tensor.float() - actual.float()).abs().max().item(),
                "previous_vs_native_max_error": (reference_tensor.float() - old.float()).abs().max().item(),
                "finite": bool(torch.isfinite(actual).all()),
            }
        passed = all(
            value["finite"] and value["graph_eager_max_error"] == 0
            and value["native_max_error"] == 0 for value in stages.values()
        )
        record = {"step": step, "stages": stages, "passed": passed}
        records.append(record)
        print(json.dumps({"routed_chain": record}), flush=True)
    return {"source_tokens": tokens, "topk": topk, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection-module")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("hy4_gmm_component", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(20260905)
    results = [run_stage(module, 6144, 128), run_stage(module, 64, 6144)]
    passed = all(r["passed"] for stage in results for r in stage["records"])
    chain = None
    if args.projection_module:
        spec = importlib.util.spec_from_file_location("hy4_projection_component", args.projection_module)
        projection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(projection)
        chain = run_routed_chain(module, projection)
        passed = passed and all(r["passed"] for r in chain["records"])
    Path(args.output).write_text(json.dumps({"passed": passed, "stages": results, "routed_chain": chain}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
