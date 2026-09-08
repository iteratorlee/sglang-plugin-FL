"""Check native BF16 lightning-indexer selection and sink DSA graph replay."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("hy4_sparse_attention", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    torch.manual_seed(20260908)
    length, page = args.context, 128
    tokens = args.queries
    if not 1 <= tokens <= length or args.steps < 0:
        parser.error("require 1 <= queries <= context and steps >= 0")
    pages = (length + page - 1) // page + 2
    q_index = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device="npu")
    index_cache = torch.randn(pages, page, 1, 128, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(tokens, 32, dtype=torch.bfloat16, device="npu")
    query = torch.randn(tokens, 2, 512, dtype=torch.bfloat16, device="npu")
    query_rope = torch.randn(tokens, 2, 64, dtype=torch.bfloat16, device="npu")
    key = torch.randn(pages, page, 1, 512, dtype=torch.bfloat16, device="npu")
    rope = torch.randn(pages, page, 1, 64, dtype=torch.bfloat16, device="npu")
    sinks = torch.tensor([-1.0, 7.0], dtype=torch.float32, device="npu")
    table = torch.randperm(pages, dtype=torch.int32, device="npu").view(1, -1)
    cuq = torch.full((1,), tokens, dtype=torch.int32, device="npu")
    lengths = torch.full((1,), length, dtype=torch.int32, device="npu")

    def run():
        indices, _ = torch_npu.npu_lightning_indexer(
            query=q_index,
            key=index_cache,
            weights=weights,
            actual_seq_lengths_query=cuq,
            actual_seq_lengths_key=lengths,
            block_table=table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=args.topk,
            sparse_mode=3,
        )
        out = module.hy4_sparse_attention(
            query, query_rope, key, rope, indices, table, cuq, lengths, sinks, 1 / 16
        )
        return indices, out

    def validate(result):
        indices, out = result
        logical = indices.reshape(tokens, -1).long().cpu()
        n = int(lengths.cpu()[0])
        kv = index_cache.float().cpu()
        pages_cpu = table.long().cpu()[0]
        qi, wi = q_index.float().cpu(), weights.float().cpu()
        qc, qr = query.float().cpu(), query_rope.float().cpu()
        kc, krc = key.float().cpu().reshape(-1, 512), rope.float().cpu().reshape(-1, 64)
        oc = out.float().cpu()
        samples = []
        for row in sorted({0, tokens // 2, tokens - 1}):
            causal_length = n - tokens + row + 1
            positions = torch.arange(causal_length)
            physical = pages_cpu[positions // page] * page + positions % page
            projected = qi[row] @ kv.reshape(-1, 128)[physical].T
            scores = (projected.relu() * wi[row, :, None]).sum(0)
            expected_ids = scores.topk(min(args.topk, causal_length)).indices
            ids = logical[row]
            valid_ids = ids[(ids >= 0) & (ids < causal_length)]
            overlap = len(set(expected_ids.tolist()) & set(valid_ids.tolist())) / len(
                expected_ids
            )
            selected = pages_cpu[valid_ids // page] * page + valid_ids % page
            k, kr = kc[selected], krc[selected]
            logits = (qc[row] @ k.T + qr[row] @ kr.T) / 16
            scores_sink = torch.cat((logits, sinks.float().cpu()[:, None]), dim=-1)
            expected = scores_sink.softmax(-1)[:, :-1] @ k
            samples.append(
                {
                    "query_row": row,
                    "causal_length": causal_length,
                    "topk_overlap_fp32": overlap,
                    "sparse_attention_reference_error": (oc[row] - expected)
                    .abs()
                    .max()
                    .item(),
                    "no_future_indices": bool(
                        ((ids < causal_length) | (ids == -1)).all()
                    ),
                }
            )
        return {
            "context": n,
            "query_tokens": tokens,
            "sparse_count": args.topk,
            "index_shape": list(indices.shape),
            "index_stride": list(indices.stride()),
            "index_dtype": str(indices.dtype),
            "index_contiguous": indices.is_contiguous(),
            "index_npu_format": torch_npu.get_npu_format(indices),
            "topk_overlap_fp32": min(s["topk_overlap_fp32"] for s in samples),
            "sparse_attention_reference_error": max(
                s["sparse_attention_reference_error"] for s in samples
            ),
            "no_future_indices": all(s["no_future_indices"] for s in samples),
            "samples": samples,
            "finite": bool(torch.isfinite(out).all().item()),
        }

    eager = run()
    torch.npu.synchronize()
    initial = validate(eager)
    print("INITIAL", json.dumps(initial), flush=True)
    started = time.perf_counter()
    run()
    torch.npu.synchronize()
    initial["warm_run_ms"] = (time.perf_counter() - started) * 1000
    if args.steps:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = run()
        graph.replay()
        torch.npu.synchronize()
    steps = []
    for step in range(args.steps):
        lengths.add_(1)
        q_index.mul_(0.93)
        weights.copy_(weights.roll(1, -1))
        table.copy_(table.roll(1, -1))
        query.mul_(0.91)
        eager_changed = run()
        graph.replay()
        torch.npu.synchronize()
        row = validate(captured)
        row["graph_index_equal"] = bool(torch.equal(captured[0], eager_changed[0]))
        row["graph_attention_error"] = (
            (captured[1].float() - eager_changed[1].float()).abs().max().item()
        )
        row["step"] = step
        steps.append(row)
        print("REPLAY", json.dumps(row), flush=True)
    passed = all(
        r["finite"]
        and r["no_future_indices"]
        and r["topk_overlap_fp32"] >= 0.98
        and r["sparse_attention_reference_error"] < 0.005
        for r in [initial, *steps]
    )
    passed &= all(
        r["graph_index_equal"] and r["graph_attention_error"] == 0 for r in steps
    )
    result = {"passed": passed, "initial": initial, "replay": steps}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
