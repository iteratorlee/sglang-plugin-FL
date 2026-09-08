#!/usr/bin/env python3
"""HYV4 graph checks for dynamic KV metadata and iHC boundaries."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch
import torch_npu

from sglang_fl.models.hy4_preview.hc import HYV4HCHeadLayer, HYV4HCLayer
from sglang_fl.models.hy4_preview.hy4_kv_cache import hy4_kv_scatter
from sglang.srt.layers.rotary_embedding.factory import get_rope_wrapper
from sglang.srt.server_args import (
    get_global_server_args,
    set_global_server_args_for_scheduler,
)


def validate_kv_scatter() -> dict:
    """Check graph writes against independent CPU pools, including padding.

    Sink attention is covered by validate_indexer_scatter_chain_npu.py using
    the production native-DSA path; no historical dense MLA kernel is needed.
    """
    torch.manual_seed(20260904)
    batch_size, pool_rows = 32, 512
    # Real key/RoPE inputs are strided slices of a shared 576-channel buffer.
    storage = torch.randn(
        batch_size, 1, 576, device="npu", dtype=torch.bfloat16
    )
    chunks = (storage[..., :512], storage[..., 512:])
    pools = [
        torch.randn(pool_rows, 1, width, device="npu", dtype=torch.bfloat16)
        for width in (512, 64)
    ]
    locations = torch.zeros(batch_size, device="npu", dtype=torch.int32)
    lengths = torch.zeros_like(locations)
    initial_pools = [pool.cpu().clone() for pool in pools]

    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        for pool, data in zip(pools, chunks):
            hy4_kv_scatter(pool, data, locations, lengths)
    graph.replay()
    torch_npu.npu.synchronize()
    padding_only_exact = all(
        torch.equal(pool.cpu(), initial)
        for pool, initial in zip(pools, initial_pools)
    )

    cases = []
    for active_rows in (1, 2, batch_size):
        # Never use an eager NPU scatter to prepare the reference pool: it
        # could hide stale captured indices. Preserve every previously written
        # row across eight launches and compare the entire cache each time.
        expected = [pool.cpu().clone() for pool in pools]
        steps = []
        for step in range(8):
            updated = torch.randn(batch_size, 1, 576, dtype=torch.bfloat16)
            new_locations = torch.zeros(batch_size, dtype=torch.int32)
            slots = 1 + step * batch_size + torch.arange(active_rows)
            new_locations[:active_rows] = slots.to(torch.int32)
            new_lengths = torch.zeros(batch_size, dtype=torch.int32)
            new_lengths[:active_rows] = step + 1
            storage.copy_(updated)
            locations.copy_(new_locations)
            lengths.copy_(new_lengths)
            for reference, data in zip(
                expected, (updated[..., :512], updated[..., 512:])
            ):
                reference[slots] = data[:active_rows]
            graph.replay()
            torch_npu.npu.synchronize()
            exact = [
                torch.equal(pool.cpu(), reference)
                for pool, reference in zip(pools, expected)
            ]
            steps.append({"step": step + 1, "pools_exact": exact, "passed": all(exact)})
        cases.append(
            {
                "active_rows": active_rows,
                "steps": steps,
                "passed": all(row["passed"] for row in steps),
            }
        )
    return {
        "input_strides": [list(data.stride()) for data in chunks],
        "padding_only_exact": padding_only_exact,
        "cases": cases,
        "passed": padding_only_exact and all(row["passed"] for row in cases),
    }


def validate_ihc() -> dict:
    torch.manual_seed(20260905)
    config = SimpleNamespace(
        enable_ihc=True,
        hidden_size=128,
        hc_mult=4,
        hc_magnitude=2.0,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
    )
    boundary = HYV4HCLayer(config, 0).to("npu")
    head = HYV4HCHeadLayer(config, 128, 4, 1e-6).to("npu")
    # ReplicatedLinear allocates loader-owned storage without initializing it;
    # component tests do not run the checkpoint loader, so seed finite weights.
    with torch.no_grad():
        boundary.hc_pre.hc_fn.weight.fill_(0.001)
        head.hc_head_fn.weight.fill_(0.001)
    x = torch.randn(32, 4, 128, device="npu", dtype=torch.bfloat16)
    block = torch.randn(32, 128, device="npu", dtype=torch.bfloat16)

    def run(value, block_value):
        reduced, post, residual = boundary.pre(value)
        updated = boundary.post(block_value + reduced * 0.01, residual, post)
        return head(updated)

    eager = run(x, block)
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_out = run(x, block)
    graph.replay()
    torch_npu.npu.synchronize()
    captured = graph_out.clone()
    initial_error = (eager.float() - captured.float()).abs().max().item()
    x.copy_(x * 0.5)
    block.copy_(block * 0.75)
    graph.replay()
    torch_npu.npu.synchronize()
    replay = graph_out.clone()
    changed_eager = run(x, block)
    torch_npu.npu.synchronize()
    changed_error = (replay.float() - changed_eager.float()).abs().max().item()
    finite = {
        "eager": bool(torch.isfinite(eager).all().item()),
        "captured": bool(torch.isfinite(captured).all().item()),
        "changed_replay": bool(torch.isfinite(replay).all().item()),
        "changed_eager": bool(torch.isfinite(changed_eager).all().item()),
    }
    return {
        "shape": list(replay.shape),
        "initial_max_abs_error": initial_error,
        "changed_input_replay_max_abs_error": changed_error,
        "finite": finite,
        "passed": (
            all(finite.values())
            and initial_error <= 0.01
            and changed_error <= 0.01
        ),
    }


def validate_native_rope() -> dict:
    """Check native NPU interleaved RoPE across dynamic graph positions."""
    torch.manual_seed(20260906)
    bs = 32
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            SimpleNamespace(rl_on_policy_target=None)
        )
    rope = get_rope_wrapper(
        64,
        rotary_dim=64,
        max_position=1024,
        base=10000,
        is_neox_style=False,
        dtype=torch.bfloat16,
        device="npu",
    ).to("npu")
    rope._forward_method = rope.forward_npu
    positions = torch.zeros(bs, dtype=torch.int64, device="npu")
    query_storage = torch.randn(
        bs, 2, 256, dtype=torch.bfloat16, device="npu"
    )
    key_storage = torch.randn(
        bs, 1, 576, dtype=torch.bfloat16, device="npu"
    )
    query = query_storage[..., 192:]
    key = key_storage[..., 512:]
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_q, graph_k = rope.forward_npu(positions, query, key)
    steps = []
    for step in range(8):
        query_storage.zero_()
        key_storage.zero_()
        q_real = torch.randn(2, 64, dtype=torch.bfloat16, device="npu")
        k_real = torch.randn(1, 64, dtype=torch.bfloat16, device="npu")
        query[:2].copy_(q_real.unsqueeze(0).expand(2, -1, -1))
        key[:2].copy_(k_real.unsqueeze(0).expand(2, -1, -1))
        positions.zero_()
        positions[:2] = step
        graph.replay()
        torch_npu.npu.synchronize()
        replay_q = graph_q.clone()
        replay_k = graph_k.clone()
        eager_q, eager_k = rope.forward_npu(positions, query, key)
        torch_npu.npu.synchronize()
        q_reference_error = (
            replay_q[:2].float() - eager_q[:2].float()
        ).abs().max().item()
        k_reference_error = (
            replay_k[:2].float() - eager_k[:2].float()
        ).abs().max().item()
        q_duplicate_error = (
            replay_q[0].float() - replay_q[1].float()
        ).abs().max().item()
        k_duplicate_error = (
            replay_k[0].float() - replay_k[1].float()
        ).abs().max().item()
        steps.append(
            {
                "step": step + 1,
                "position": step,
                "q_graph_vs_eager_max_abs_error": q_reference_error,
                "k_graph_vs_eager_max_abs_error": k_reference_error,
                "q_duplicate_lane_max_abs_error": q_duplicate_error,
                "k_duplicate_lane_max_abs_error": k_duplicate_error,
                "passed": q_reference_error <= 0.01
                and k_reference_error <= 0.01
                and q_duplicate_error == 0.0
                and k_duplicate_error == 0.0,
            }
        )
    return {
        "query_stride": list(query.stride()),
        "key_stride": list(key.stride()),
        "steps": steps,
        "passed": all(row["passed"] for row in steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {
        "kv_scatter": validate_kv_scatter(),
        "ihc": validate_ihc(),
        "native_rope": validate_native_rope(),
    }
    result["passed"] = (
        result["kv_scatter"]["passed"]
        and result["ihc"]["passed"]
        and result["native_rope"]["passed"]
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
