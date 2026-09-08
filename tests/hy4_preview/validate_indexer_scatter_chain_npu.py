"""Validate HYV4's graph-dynamic scatter -> indexer -> sparse-attention chain.

The graph and eager paths use independent cache tensors.  This is deliberate:
running eager on the graph's cache before replay could make a stale captured
scatter look correct.  The test changes every dynamic input across replays and
checks both cache integrity and numerical results at 8K and 128K contexts.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu


PAGE_SIZE = 128
INDEX_DIM = 128
KV_DIM = 512
ROPE_DIM = 64
INDEX_HEADS = 32
ATTENTION_HEADS = 2
TOPK = 2048


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _physical_positions(table, logical_positions):
    return (
        table[logical_positions // PAGE_SIZE] * PAGE_SIZE
        + logical_positions % PAGE_SIZE
    )


def _cpu_reference(
    indices,
    attention_output,
    index_query,
    index_weights,
    index_cache,
    query,
    query_rope,
    key_cache,
    rope_cache,
    table,
    length,
    sinks,
):
    table_cpu = table[0].long().cpu()
    logical_positions = torch.arange(length, dtype=torch.int64)
    physical = _physical_positions(table_cpu, logical_positions)

    index_keys = index_cache.reshape(-1, INDEX_DIM).float().cpu()[physical]
    index_scores = torch.matmul(index_query[0].float().cpu(), index_keys.T)
    index_scores = (
        index_scores.relu() * index_weights[0].float().cpu().unsqueeze(1)
    ).sum(0)
    expected_topk = index_scores.topk(min(TOPK, length)).indices

    logical = indices.reshape(-1).long().cpu()
    valid = logical[(logical >= 0) & (logical < length)]
    overlap = len(set(expected_topk.tolist()) & set(valid.tolist())) / len(
        expected_topk
    )
    selected_physical = _physical_positions(table_cpu, valid)
    selected_key = key_cache.reshape(-1, KV_DIM).float().cpu()[selected_physical]
    selected_rope = (
        rope_cache.reshape(-1, ROPE_DIM).float().cpu()[selected_physical]
    )
    logits = (
        torch.matmul(query[0].float().cpu(), selected_key.T)
        + torch.matmul(query_rope[0].float().cpu(), selected_rope.T)
    ) / 16
    logits_with_sink = torch.cat(
        (logits, sinks.float().cpu().unsqueeze(1)), dim=-1
    )
    expected_attention = (
        logits_with_sink.softmax(-1)[:, :-1] @ selected_key
    ).unsqueeze(0)
    return {
        "topk_overlap_fp32": overlap,
        "newest_logical_selected": bool((valid == length - 1).any()),
        "no_future_indices": bool(((logical < length) | (logical == -1)).all()),
        "sparse_attention_reference_error": (
            attention_output.float().cpu() - expected_attention
        )
        .abs()
        .max()
        .item(),
    }


def _cache_rows_equal(cache, expected_rows):
    flat = cache.reshape(-1, cache.shape[-1])
    return all(
        torch.equal(flat[slot].cpu(), expected) for slot, expected in expected_rows.items()
    )


def _set_step_inputs(
    step,
    base_table,
    table,
    length_tensor,
    scatter_mask,
    locations,
    index_data,
    key_data,
    rope_data,
    index_query,
    index_weights,
    query,
    query_rope,
    sinks,
    base_context,
):
    length = base_context + step
    table_cpu = base_table.roll(step)
    table.copy_(table_cpu.view(1, -1).to(torch.int32).npu())
    length_tensor.fill_(length)
    scatter_mask.copy_(torch.tensor([length, 0], dtype=torch.int32, device="npu"))
    logical = length - 1
    target = int(
        table_cpu[logical // PAGE_SIZE] * PAGE_SIZE + logical % PAGE_SIZE
    )
    locations.copy_(torch.tensor([target, 0], dtype=torch.int64, device="npu"))

    index_pattern = torch.linspace(
        0.5, 1.5, INDEX_DIM, dtype=torch.float32, device="npu"
    ).roll(step)
    head_offsets = torch.linspace(
        0, 0.031, INDEX_HEADS, dtype=torch.float32, device="npu"
    ).unsqueeze(1)
    index_query.copy_(
        (index_pattern.unsqueeze(0) + head_offsets)
        .unsqueeze(0)
        .to(torch.bfloat16)
    )
    index_weights.copy_(
        torch.linspace(
            0.5, 1.5, INDEX_HEADS, dtype=torch.float32, device="npu"
        )
        .roll(step)
        .unsqueeze(0)
        .to(torch.bfloat16)
    )
    index_data[0].copy_((index_pattern * (2.0 + step / 8)).to(torch.bfloat16))
    index_data[1].fill_(77 + step)

    torch.manual_seed(20260910 + step)
    key_data[0].copy_(torch.randn_like(key_data[0]) * 0.25 + step / 16)
    rope_data[0].copy_(torch.randn_like(rope_data[0]) * 0.25 - step / 32)
    key_data[1].fill_(55 + step)
    rope_data[1].fill_(33 + step)
    query.copy_(torch.randn_like(query) * 0.2 + step / 64)
    query_rope.copy_(torch.randn_like(query_rope) * 0.2 - step / 64)
    sinks.copy_(
        torch.tensor([-1.0 + step / 10, 7.0 - step / 10], device="npu")
    )
    return length, target, table_cpu


def _validate_case(context, steps, scatter_module, attention_module):
    logical_pages = (context + steps + PAGE_SIZE - 1) // PAGE_SIZE
    pool_pages = logical_pages + 8
    generator = torch.Generator().manual_seed(20260909 + context)
    base_table = torch.randperm(logical_pages, generator=generator).long() + 1

    index_base = (
        torch.randn(
            pool_pages,
            PAGE_SIZE,
            1,
            INDEX_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        * 0.01
    )
    key_base = (
        torch.randn(
            pool_pages,
            PAGE_SIZE,
            1,
            KV_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        * 0.05
    )
    rope_base = (
        torch.randn(
            pool_pages,
            PAGE_SIZE,
            1,
            ROPE_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        * 0.05
    )
    graph_caches = [index_base, key_base, rope_base]
    eager_caches = [item.clone() for item in graph_caches]
    slot_zero = [item.reshape(-1, item.shape[-1])[0].cpu().clone() for item in graph_caches]

    table_graph = torch.empty(
        1, logical_pages, dtype=torch.int32, device="npu"
    )
    table_eager = torch.empty_like(table_graph)
    lengths = torch.empty(1, dtype=torch.int32, device="npu")
    scatter_mask = torch.empty(2, dtype=torch.int32, device="npu")
    locations = torch.empty(2, dtype=torch.int64, device="npu")
    index_data = torch.empty(2, INDEX_DIM, dtype=torch.bfloat16, device="npu")
    key_data = torch.empty(2, 1, KV_DIM, dtype=torch.bfloat16, device="npu")
    rope_data = torch.empty(2, 1, ROPE_DIM, dtype=torch.bfloat16, device="npu")
    index_query = torch.empty(
        1, INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16, device="npu"
    )
    index_weights = torch.empty(
        1, INDEX_HEADS, dtype=torch.bfloat16, device="npu"
    )
    query = torch.empty(
        1, ATTENTION_HEADS, KV_DIM, dtype=torch.bfloat16, device="npu"
    )
    query_rope = torch.empty(
        1, ATTENTION_HEADS, ROPE_DIM, dtype=torch.bfloat16, device="npu"
    )
    sinks = torch.empty(ATTENTION_HEADS, dtype=torch.float32, device="npu")
    cuq = torch.ones(1, dtype=torch.int32, device="npu")

    def chain(caches, table):
        index_cache, key_cache, rope_cache = caches
        scatter_module.hy4_kv_scatter(
            index_cache, index_data, locations, scatter_mask
        )
        scatter_module.hy4_kv_scatter(key_cache, key_data, locations, scatter_mask)
        scatter_module.hy4_kv_scatter(rope_cache, rope_data, locations, scatter_mask)
        indices, _ = torch_npu.npu_lightning_indexer(
            query=index_query,
            key=index_cache,
            weights=index_weights,
            actual_seq_lengths_query=cuq,
            actual_seq_lengths_key=lengths,
            block_table=table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )
        attention = attention_module.hy4_sparse_attention(
            query,
            query_rope,
            key_cache,
            rope_cache,
            indices,
            table,
            cuq,
            lengths,
            sinks,
            1 / 16,
        )
        return indices, attention

    written_index = {}
    written_key = {}
    written_rope = {}
    targets = []

    def validate(step, graph_result, eager_result, length, target, table_cpu):
        graph_indices, graph_attention = graph_result
        eager_indices, eager_attention = eager_result
        newest = {
            target: index_data[0].cpu().clone(),
        }
        written_index.update(newest)
        written_key[target] = key_data[0].reshape(-1).cpu().clone()
        written_rope[target] = rope_data[0].reshape(-1).cpu().clone()
        targets.append(target)
        reference = _cpu_reference(
            graph_indices,
            graph_attention,
            index_query,
            index_weights,
            graph_caches[0],
            query,
            query_rope,
            graph_caches[1],
            graph_caches[2],
            table_graph,
            length,
            sinks,
        )
        row = {
            "step": step,
            "context": length,
            "target_slot": target,
            "target_slots_unique": len(set(targets)) == len(targets),
            "table_matches_step": bool(
                torch.equal(table_graph[0].cpu(), table_cpu.to(torch.int32))
            ),
            "graph_index_equal_eager": bool(
                torch.equal(graph_indices, eager_indices)
            ),
            "graph_attention_error_eager": (
                graph_attention.float() - eager_attention.float()
            )
            .abs()
            .max()
            .item(),
            "graph_caches_equal_eager": all(
                torch.equal(graph_cache, eager_cache)
                for graph_cache, eager_cache in zip(graph_caches, eager_caches)
            ),
            "target_rows_exact": (
                _cache_rows_equal(graph_caches[0], {target: written_index[target]})
                and _cache_rows_equal(graph_caches[1], {target: written_key[target]})
                and _cache_rows_equal(graph_caches[2], {target: written_rope[target]})
            ),
            "old_slots_preserved": (
                _cache_rows_equal(graph_caches[0], written_index)
                and _cache_rows_equal(graph_caches[1], written_key)
                and _cache_rows_equal(graph_caches[2], written_rope)
            ),
            "slot_zero_preserved": all(
                torch.equal(cache.reshape(-1, cache.shape[-1])[0].cpu(), expected)
                for cache, expected in zip(graph_caches, slot_zero)
            ),
            "finite": bool(torch.isfinite(graph_attention).all().item()),
            **reference,
        }
        row["passed"] = (
            row["target_slots_unique"]
            and row["table_matches_step"]
            and row["graph_index_equal_eager"]
            and row["graph_attention_error_eager"] == 0
            and row["graph_caches_equal_eager"]
            and row["target_rows_exact"]
            and row["old_slots_preserved"]
            and row["slot_zero_preserved"]
            and row["finite"]
            and row["topk_overlap_fp32"] >= 0.98
            and row["newest_logical_selected"]
            and row["no_future_indices"]
            and row["sparse_attention_reference_error"] < 0.005
        )
        print(json.dumps(row), flush=True)
        return row

    length, target, table_cpu = _set_step_inputs(
        0,
        base_table,
        table_graph,
        lengths,
        scatter_mask,
        locations,
        index_data,
        key_data,
        rope_data,
        index_query,
        index_weights,
        query,
        query_rope,
        sinks,
        context,
    )
    table_eager.copy_(table_graph)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = chain(graph_caches, table_graph)
    # Run eager only on its independent cache, then replay last.  Some CANN
    # operators reuse process-global workspaces, so evaluating eager after the
    # replay could overwrite a captured result even though its cache is separate.
    eager_initial = chain(eager_caches, table_eager)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    rows = [
        validate(0, captured, eager_initial, length, target, table_cpu)
    ]

    for step in range(1, steps + 1):
        length, target, table_cpu = _set_step_inputs(
            step,
            base_table,
            table_graph,
            lengths,
            scatter_mask,
            locations,
            index_data,
            key_data,
            rope_data,
            index_query,
            index_weights,
            query,
            query_rope,
            sinks,
            context,
        )
        table_eager.copy_(table_graph)
        eager_result = chain(eager_caches, table_eager)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        rows.append(
            validate(step, captured, eager_result, length, target, table_cpu)
        )

    result = {
        "base_context": context,
        "replay_steps": steps,
        "page_size": PAGE_SIZE,
        "topk": TOPK,
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    source_dir = (
        Path(__file__).resolve().parents[2]
        / "sglang_fl"
        / "models"
        / "hy4_preview"
    )
    parser.add_argument("--scatter-module", type=Path, default=source_dir / "hy4_kv_cache.py")
    parser.add_argument(
        "--attention-module", type=Path, default=source_dir / "hy4_sparse_attention.py"
    )
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 131072])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 4:
        parser.error("--steps must be at least 4")

    torch.npu.set_device(0)
    torch.manual_seed(20260909)
    scatter_module = _load_module("hy4_chain_scatter", args.scatter_module)
    attention_module = _load_module("hy4_chain_attention", args.attention_module)
    cases = [
        _validate_case(context, args.steps, scatter_module, attention_module)
        for context in args.contexts
    ]
    result = {"passed": all(case["passed"] for case in cases), "cases": cases}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
