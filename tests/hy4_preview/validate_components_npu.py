#!/usr/bin/env python3
"""HYV4 graph checks for dynamic KV metadata and iHC boundaries."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch
import torch_npu

from sglang_fl.models.hy4_preview.hc import HYV4HCHeadLayer, HYV4HCLayer
from sglang_fl.models.hy4_preview.hy4_triton_attn import (
    hy4_kv_scatter,
    hy4_mla_decode_sinks_triton,
)
from sglang_fl.models.hy4_preview.hy4_triton_projection import hy4_head_bmm
from sglang.srt.layers.rotary_embedding.factory import get_rope_wrapper
from sglang.srt.server_args import (
    get_global_server_args,
    set_global_server_args_for_scheduler,
)


def decode_chain(qn, qr, kn, kr, key_pool, rope_pool, loc, sinks, r2t, req, lens, wv, gate):
    hy4_kv_scatter(key_pool, kn, loc, lens)
    hy4_kv_scatter(rope_pool, kr, loc, lens)
    attn = hy4_mla_decode_sinks_triton(
        qn, qr, key_pool, rope_pool, sinks, r2t, req, lens,
        1.0 / (576.0**0.5), 2, 512, 64,
    )
    return hy4_head_bmm(attn, wv, gate)


def decode_reference(qn, qr, key_pool, rope_pool, sinks, r2t, req, lens, wv, gate):
    """Small FP32 definition of sink decode plus the gated V projection."""
    rows = []
    scale = 1.0 / (576.0**0.5)
    for index in range(qn.shape[0]):
        length = int(lens[index].item())
        if length == 0:
            rows.append(torch.zeros_like(qn[index]))
            continue
        token_ids = r2t[req[index].long(), :length].long()
        key = key_pool[token_ids, 0].float()
        rope = rope_pool[token_ids, 0].float()
        score = (
            qn[index].float() @ key.transpose(0, 1)
            + qr[index].float() @ rope.transpose(0, 1)
        ) * scale
        score_with_sink = torch.cat((score, sinks[:, None].float()), dim=-1)
        prob = torch.softmax(score_with_sink, dim=-1)[:, :length]
        rows.append((prob @ key).to(torch.bfloat16))
    attn = torch.stack(rows)
    projected = torch.einsum("thk,hkn->thn", attn.float(), wv.float())
    return (projected * gate.view_as(projected).float()).to(torch.bfloat16)


def validate_decode_chain() -> dict:
    torch.manual_seed(20260904)
    bs, pool_rows, max_ctx = 32, 256, 8
    qn = torch.randn(bs, 2, 512, device="npu", dtype=torch.bfloat16) * 0.02
    q_storage = (
        torch.randn(bs, 2, 256, device="npu", dtype=torch.bfloat16) * 0.02
    )
    qr = q_storage[..., 192:]
    kn = torch.randn(bs, 1, 512, device="npu", dtype=torch.bfloat16) * 0.02
    kv_storage = (
        torch.randn(bs, 1, 576, device="npu", dtype=torch.bfloat16) * 0.02
    )
    kr = kv_storage[..., 512:]
    qn_base, qr_base = qn.clone(), qr.clone()
    kn_base, kr_base = kn.clone(), kr.clone()
    key_base = torch.randn(pool_rows, 1, 512, device="npu", dtype=torch.bfloat16) * 0.02
    rope_base = torch.randn(pool_rows, 1, 64, device="npu", dtype=torch.bfloat16) * 0.02
    key_pool, rope_pool = key_base.clone(), rope_base.clone()
    loc = torch.zeros(bs, device="npu", dtype=torch.int32)
    sinks = torch.randn(2, device="npu", dtype=torch.float32)
    r2t = torch.zeros(bs, max_ctx, device="npu", dtype=torch.int32)
    req = torch.arange(bs, device="npu", dtype=torch.int32)
    lens = torch.zeros(bs, device="npu", dtype=torch.int32)
    wv = torch.randn(2, 512, 256, device="npu", dtype=torch.bfloat16) * 0.02
    gate = torch.sigmoid(
        torch.randn(bs, 2 * 256, device="npu", dtype=torch.bfloat16)
    )
    gate_base = gate.clone()

    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        graph_out = decode_chain(
            qn, qr, kn, kr, key_pool, rope_pool, loc, sinks, r2t, req, lens, wv, gate
        )

    scenarios = []
    multi_loc = torch.arange(96, 96 + bs, device="npu", dtype=torch.int32)
    multi_lens_cpu = torch.tensor(
        [(index % 4) + 1 for index in range(bs)], dtype=torch.int32
    )
    multi_r2t_cpu = torch.stack(
        [torch.arange(index * max_ctx, (index + 1) * max_ctx, dtype=torch.int32) for index in range(bs)]
    )
    # The newest valid token of every request must resolve to the location
    # written by the captured KV scatter.  Earlier positions stay in distinct
    # pre-populated base rows.
    for index in range(bs):
        multi_r2t_cpu[index, int(multi_lens_cpu[index]) - 1] = 96 + index
    raw1_loc = torch.tensor(
        [96] + [0] * (bs - 1), device="npu", dtype=torch.int32
    )
    raw1_r2t = torch.zeros((bs, max_ctx), device="npu", dtype=torch.int32)
    raw1_r2t[0, 0] = 96
    for name, new_loc, new_lens, new_r2t in (
        (
            "raw1_with_padding",
            raw1_loc,
            torch.tensor([1] + [0] * (bs - 1), device="npu", dtype=torch.int32),
            raw1_r2t,
        ),
        (
            "multi_request_changed_loc",
            multi_loc,
            multi_lens_cpu.to("npu"),
            multi_r2t_cpu.to("npu"),
        ),
    ):
        scale = 0.5 if name.startswith("raw1") else 0.75
        replay_qn = qn_base * scale
        replay_qr = qr_base * (scale + 0.1)
        replay_kn = kn_base * (scale + 0.2)
        replay_kr = kr_base * (scale + 0.3)
        replay_gate = torch.sigmoid(gate.float() * 0.25).to(torch.bfloat16)
        key_pool.copy_(key_base)
        rope_pool.copy_(rope_base)
        qn.copy_(replay_qn)
        qr.copy_(replay_qr)
        kn.copy_(replay_kn)
        kr.copy_(replay_kr)
        gate.copy_(replay_gate)
        loc.copy_(new_loc)
        lens.copy_(new_lens)
        r2t.copy_(new_r2t)
        graph.replay()
        torch_npu.npu.synchronize()
        replay = graph_out.clone()
        active = new_lens > 0
        active_loc = new_loc[active]
        key_scatter_error = (
            key_pool[active_loc.long(), 0].float() - kn[active, 0].float()
        ).abs().max().item()
        rope_scatter_error = (
            rope_pool[active_loc.long(), 0].float() - kr[active, 0].float()
        ).abs().max().item()
        padding_slot_error = (
            key_pool[0].float() - key_base[0].float()
        ).abs().max().item()
        rope_padding_slot_error = (
            rope_pool[0].float() - rope_base[0].float()
        ).abs().max().item()
        reference = decode_reference(
            qn, qr, key_pool, rope_pool, sinks, new_r2t, req, new_lens, wv, replay_gate
        )
        reference_error = (
            replay.float() - reference.float()
        ).abs().max().item()

        eager = decode_chain(
            qn,
            qr,
            kn,
            kr,
            key_base.clone(),
            rope_base.clone(),
            new_loc,
            sinks,
            new_r2t,
            req,
            new_lens,
            wv,
            replay_gate,
        )
        torch_npu.npu.synchronize()
        error = (replay.float() - eager.float()).abs().max().item()
        scenarios.append(
            {
                "name": name,
                "max_abs_error": error,
                "fp32_reference_max_abs_error": reference_error,
                "key_scatter_max_abs_error": key_scatter_error,
                "rope_scatter_max_abs_error": rope_scatter_error,
                "padding_slot0_max_abs_error": padding_slot_error,
                "rope_padding_slot0_max_abs_error": rope_padding_slot_error,
                "passed": (
                    error <= 0.01
                    and reference_error <= 0.125
                    and key_scatter_error == 0.0
                    and rope_scatter_error == 0.0
                    and padding_slot_error == 0.0
                    and rope_padding_slot_error == 0.0
                ),
            }
        )
    # Real decode accumulates one new KV row per replay.  Preserve the pool
    # across three graph launches, change loc/seq_len/query/KV each step, and
    # compare every output against the FP32 definition using the full history.
    key_pool.copy_(key_base)
    rope_pool.copy_(rope_base)
    loc.zero_()
    lens.zero_()
    r2t.zero_()
    cumulative = []
    saved_key_rows = []
    saved_rope_rows = []
    for step in range(3):
        factor = 0.6 + step * 0.1
        qn.copy_(qn_base * factor)
        qr.copy_(qr_base * (factor + 0.05))
        kn.copy_(kn_base * (factor + 0.1))
        kr.copy_(kr_base * (factor + 0.15))
        gate.copy_(torch.sigmoid(gate_base.float() * factor).to(torch.bfloat16))
        loc.zero_()
        loc[0] = 160 + step
        lens.zero_()
        lens[0] = step + 1
        r2t[0, step] = 160 + step
        graph.replay()
        torch_npu.npu.synchronize()
        replay = graph_out.clone()
        saved_key_rows.append(kn[0, 0].clone())
        saved_rope_rows.append(kr[0, 0].clone())
        reference = decode_reference(
            qn, qr, key_pool, rope_pool, sinks, r2t, req, lens, wv, gate
        )
        reference_error = (
            replay.float() - reference.float()
        ).abs().max().item()
        key_history_error = max(
            (
                key_pool[160 + index, 0].float() - expected.float()
            ).abs().max().item()
            for index, expected in enumerate(saved_key_rows)
        )
        rope_history_error = max(
            (
                rope_pool[160 + index, 0].float() - expected.float()
            ).abs().max().item()
            for index, expected in enumerate(saved_rope_rows)
        )
        slot0_error = (
            key_pool[0].float() - key_base[0].float()
        ).abs().max().item()
        cumulative.append(
            {
                "step": step + 1,
                "seq_len": step + 1,
                "loc": 160 + step,
                "fp32_reference_max_abs_error": reference_error,
                "key_history_max_abs_error": key_history_error,
                "rope_history_max_abs_error": rope_history_error,
                "padding_slot0_max_abs_error": slot0_error,
                "passed": reference_error <= 0.125
                and key_history_error == 0.0
                and rope_history_error == 0.0
                and slot0_error == 0.0,
            }
        )

    # Model-level packed BS2 first diverged only after six equal decode tokens.
    # Reproduce the exact state shape here: two active, identical requests plus
    # 30 graph-padding lanes, eight successive dynamic loc/seq_len replays, and
    # disjoint request-to-token rows.  Each active lane must match both the
    # FP32 definition and its duplicate peer at every step.
    key_pool.copy_(key_base)
    rope_pool.copy_(rope_base)
    loc.zero_()
    lens.zero_()
    r2t.zero_()
    cumulative_bs2 = []
    for step in range(8):
        factor = 0.55 + step * 0.05
        qn_step = qn_base[0] * factor
        qr_step = qr_base[0] * (factor + 0.03)
        kn_step = kn_base[0] * (factor + 0.07)
        kr_step = kr_base[0] * (factor + 0.11)
        gate_step = torch.sigmoid(
            gate_base[0].float() * factor
        ).to(torch.bfloat16)
        qn.zero_()
        qr.zero_()
        kn.zero_()
        kr.zero_()
        gate.zero_()
        qn[:2].copy_(qn_step.unsqueeze(0).expand(2, -1, -1))
        qr[:2].copy_(qr_step.unsqueeze(0).expand(2, -1, -1))
        kn[:2].copy_(kn_step.unsqueeze(0).expand(2, -1, -1))
        kr[:2].copy_(kr_step.unsqueeze(0).expand(2, -1, -1))
        gate[:2].copy_(gate_step.unsqueeze(0).expand(2, -1))
        loc.zero_()
        loc[0] = 160 + step
        loc[1] = 192 + step
        lens.zero_()
        lens[:2] = step + 1
        r2t[0, step] = 160 + step
        r2t[1, step] = 192 + step
        graph.replay()
        torch_npu.npu.synchronize()
        replay = graph_out.clone()
        reference = decode_reference(
            qn, qr, key_pool, rope_pool, sinks, r2t, req, lens, wv, gate
        )
        reference_error = (
            replay[:2].float() - reference[:2].float()
        ).abs().max().item()
        duplicate_error = (
            replay[0].float() - replay[1].float()
        ).abs().max().item()
        key_history_error = max(
            (
                key_pool[160 + index, 0].float()
                - key_pool[192 + index, 0].float()
            ).abs().max().item()
            for index in range(step + 1)
        )
        rope_history_error = max(
            (
                rope_pool[160 + index, 0].float()
                - rope_pool[192 + index, 0].float()
            ).abs().max().item()
            for index in range(step + 1)
        )
        slot0_error = (
            key_pool[0].float() - key_base[0].float()
        ).abs().max().item()
        cumulative_bs2.append(
            {
                "step": step + 1,
                "seq_len": step + 1,
                "locs": [160 + step, 192 + step],
                "fp32_reference_max_abs_error": reference_error,
                "duplicate_lane_max_abs_error": duplicate_error,
                "key_duplicate_history_max_abs_error": key_history_error,
                "rope_duplicate_history_max_abs_error": rope_history_error,
                "padding_slot0_max_abs_error": slot0_error,
                "passed": reference_error <= 0.125
                and duplicate_error == 0.0
                and key_history_error == 0.0
                and rope_history_error == 0.0
                and slot0_error == 0.0,
            }
        )
    return {
        "passed": all(row["passed"] for row in scenarios)
        and all(row["passed"] for row in cumulative)
        and all(row["passed"] for row in cumulative_bs2),
        "q_rope_stride": list(qr.stride()),
        "q_rope_contiguous": qr.is_contiguous(),
        "k_rope_stride": list(kr.stride()),
        "k_rope_contiguous": kr.is_contiguous(),
        "cases": scenarios,
        "cumulative_decode_replays": cumulative,
        "cumulative_bs2_decode_replays": cumulative_bs2,
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
        "decode_chain": validate_decode_chain(),
        "ihc": validate_ihc(),
        "native_rope": validate_native_rope(),
    }
    result["passed"] = (
        result["decode_chain"]["passed"]
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
