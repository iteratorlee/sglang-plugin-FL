"""Validate real TP32 DSA shapes, paged causality, sinks and graph replay."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def reference(q, qr, key, rope, indices, table, cuq, lengths, sinks):
    q, qr, key, rope = [x.float().cpu() for x in (q, qr, key, rope)]
    indices, table, cuq, lengths = [x.cpu() for x in (indices, table, cuq, lengths)]
    sinks = sinks.float().cpu()
    result = torch.zeros_like(q)
    start = 0
    for batch, end in enumerate(cuq.tolist()):
        for row in range(start, end):
            causal_end = int(lengths[batch]) - (end - start) + row - start + 1
            logical = indices[row].flatten().long()
            logical = logical[(logical >= 0) & (logical < causal_end)]
            if not logical.numel():
                continue
            physical = table[batch, logical // key.shape[1]].long() * key.shape[1]
            physical += logical % key.shape[1]
            selected = key.view(-1, key.shape[-1])[physical]
            selected_rope = rope.view(-1, rope.shape[-1])[physical]
            logits = (q[row] @ selected.T + qr[row] @ selected_rope.T) / 16
            logits_sink = torch.cat((logits, sinks[:, None]), dim=-1)
            result[row] = logits_sink.softmax(-1)[:, :-1] @ selected
        start = end
    return result


def case(run_attention, query_lengths, kv_lengths, graph_mode, padding=0):
    torch.manual_seed(719 + sum(query_lengths))
    page, heads, dn, dr, topk = 128, 2, 512, 64, 2048
    t = sum(query_lengths) + padding
    max_len = max(kv_lengths)
    logical_pages = (max_len + page - 1) // page
    pages = logical_pages * len(kv_lengths) + 3
    key = (torch.randn(pages, page, 1, dn) * 0.5).bfloat16().npu()
    rope = (torch.randn(pages, page, 1, dr) * 0.5).bfloat16().npu()
    q_storage = (torch.randn(t, heads, dn + dr) * 0.5).bfloat16().npu()
    q, qr = q_storage.split((dn, dr), -1)
    table_cpu = torch.randperm(pages)[: logical_pages * len(kv_lengths)]
    table = table_cpu.reshape(len(kv_lengths), logical_pages).int().npu()
    indices_cpu = torch.full((t, topk), -1, dtype=torch.int32)
    begin = 0
    for qlen, kvlen in zip(query_lengths, kv_lengths):
        for row in range(begin, begin + qlen):
            # Deliberately include later logical positions; the wrapper must
            # apply each query's causal boundary before mapping physical pages.
            selected = torch.randperm(kvlen)[: min(topk, kvlen)]
            indices_cpu[row, : selected.numel()] = selected.int()
        begin += qlen
    indices = indices_cpu.npu()
    cuq = torch.tensor(query_lengths, dtype=torch.int32).cumsum(0).int().npu()
    lengths = torch.tensor(kv_lengths, dtype=torch.int32).npu()
    sinks = torch.tensor([-3.0, 9.0], dtype=torch.float32, device="npu")

    def run():
        return run_attention(
            q, qr, key, rope, indices, table, cuq, lengths, sinks, 1 / 16
        )

    expected = reference(q, qr, key, rope, indices, table, cuq, lengths, sinks)
    output = run()
    torch.npu.synchronize()
    finite = bool(torch.isfinite(output).all().item())
    eager_error = (output.float().cpu() - expected).abs().max().item()
    row = {
        "query_lengths": query_lengths,
        "kv_lengths": kv_lengths,
        "padding_queries": padding,
        "finite": finite,
        "eager_reference_error": eager_error,
    }
    if graph_mode:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = run()
        graph.replay()
        torch.npu.synchronize()
        row["graph_eager_error"] = (
            (captured.float() - output.float()).abs().max().item()
        )
        # Alter addresses, chosen keys, history lengths and Q activations while
        # retaining every graph input's storage address.
        table.copy_(table.flip(-1))
        indices.copy_(torch.roll(indices, 61, -1))
        indices[:, ::7] = -1
        lengths.sub_(3)
        q_storage.mul_(0.7)
        sinks.add_(1)
        expected_changed = reference(
            q, qr, key, rope, indices, table, cuq, lengths, sinks
        )
        eager_changed = run()
        graph.replay()
        torch.npu.synchronize()
        row["changed_finite"] = bool(torch.isfinite(captured).all().item())
        row["changed_graph_eager_error"] = (
            (captured.float() - eager_changed.float()).abs().max().item()
        )
        row["changed_reference_error"] = (
            (captured.float().cpu() - expected_changed).abs().max().item()
        )
    row["passed"] = finite and eager_error < 0.005
    if graph_mode:
        row["passed"] &= (
            row["changed_finite"] and row["changed_reference_error"] < 0.005
        )
        row["passed"] &= (
            row["graph_eager_error"] == 0 and row["changed_graph_eager_error"] == 0
        )
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        default=str(
            Path(__file__).resolve().parents[2]
            / "sglang_fl/models/hy4_preview/hy4_sparse_attention.py"
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("hy4_sparse_attention", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    cases = [
        case(module.hy4_sparse_attention, q, k, graph, padding)
        for q, k, graph, padding in [
            ([1], [4096], True, 0),
            ([37], [8192], True, 0),
            ([3, 5], [100, 4096], True, 0),
            ([33], [32768], False, 0),
            ([1], [4096], True, 127),
            ([1], [131072], True, 0),
        ]
    ]
    result = {"passed": all(row["passed"] for row in cases), "cases": cases}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
