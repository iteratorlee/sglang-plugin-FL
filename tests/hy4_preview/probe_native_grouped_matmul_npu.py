"""Probe native Ascend INT8 grouped matmul for HYV4 prefill.

This is intentionally a stand-alone component probe.  It uses the rank-local
HYV4 expert shapes and starts from the plugin's retained ND ``[E, N, K]``
weights.  It does not import or mutate the model implementation.

Run in a container with exactly one visible NPU, for example::

    ASCEND_RT_VISIBLE_DEVICES=2 python \
      tests/hy4_preview/probe_native_grouped_matmul_npu.py \
      --tokens 512 1024 4096 --output /tmp/hy4_native_gmm.json
"""

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch
import torch_npu


def _synchronize():
    torch.npu.synchronize()


def _storage_bytes(tensor):
    return tensor.untyped_storage().nbytes()


def _counts_with_empty_experts(tokens, experts):
    """Return sorted-routing counts while deliberately leaving empty experts."""
    active = torch.arange(0, experts, 3, dtype=torch.int64)
    counts = torch.zeros(experts, dtype=torch.int64)
    assignments = active[torch.arange(tokens, dtype=torch.int64) % active.numel()]
    counts.scatter_add_(0, assignments, torch.ones_like(assignments))
    return counts


def _sample_rows(tokens, limit=16):
    if tokens <= limit:
        return torch.arange(tokens, dtype=torch.int64)
    return torch.linspace(0, tokens - 1, steps=limit).round().to(torch.int64).unique()


def _native_gmm(x_q, weight_ekn, weight_scale, token_scale, group_list, list_type):
    return torch.ops.npu.npu_grouped_matmul(
        x=[x_q],
        weight=[weight_ekn],
        scale=[weight_scale],
        per_token_scale=[token_scale],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=list_type,
        output_dtype=torch.bfloat16,
    )[0]


def _reference_sample(x_q, weight_enk, weight_scale, token_scale, counts, rows, outputs):
    expert_for_row = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.int64), counts
    )
    selected_experts = expert_for_row[rows].to(device=weight_enk.device)
    rows_npu = rows.to(device=x_q.device)
    output_ids = torch.linspace(
        0, weight_enk.shape[1] - 1, steps=min(outputs, weight_enk.shape[1])
    ).round().to(torch.int64).unique()
    output_ids_npu = output_ids.to(device=weight_enk.device)
    result = []
    for row_npu, expert_npu in zip(rows_npu, selected_experts):
        weights = weight_enk[expert_npu, output_ids_npu].float()
        accum = torch.matmul(weights, x_q[row_npu].float())
        result.append(
            accum
            * token_scale[row_npu].float()
            * weight_scale[expert_npu, output_ids_npu].float()
        )
    return torch.stack(result), rows_npu, output_ids_npu


def _compare_sample(actual, reference, rows, outputs):
    sampled = actual[rows][:, outputs].float()
    diff = (sampled - reference).abs()
    scale = reference.abs().max().item()
    max_abs = diff.max().item()
    return {
        "sample_rows": rows.cpu().tolist(),
        "sample_outputs": outputs.cpu().tolist(),
        "max_abs_error": max_abs,
        "mean_abs_error": diff.mean().item(),
        "reference_max_abs": scale,
        "normalized_max_error": max_abs / max(scale, 1e-12),
        "allclose_atol_2e-2_rtol_3e-2": bool(
            torch.allclose(sampled, reference, atol=2e-2, rtol=3e-2)
        ),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def _time_call(call, warmup, repeats):
    for _ in range(warmup):
        call()
    _synchronize()
    started = time.perf_counter()
    output = None
    for _ in range(repeats):
        output = call()
    _synchronize()
    return output, (time.perf_counter() - started) * 1000 / repeats


def _probe_layout(
    name,
    weight_ekn,
    x_q,
    weight_scale,
    token_scale,
    group_list,
    list_type,
    reference,
    rows,
    outputs,
    warmup,
    repeats,
):
    record = {
        "name": name,
        "shape": list(weight_ekn.shape),
        "stride": list(weight_ekn.stride()),
        "contiguous": weight_ekn.is_contiguous(),
        "npu_format": str(torch_npu.get_npu_format(weight_ekn)),
        "storage_bytes": _storage_bytes(weight_ekn),
    }
    try:
        call = lambda: _native_gmm(
            x_q,
            weight_ekn,
            weight_scale,
            token_scale,
            group_list,
            list_type,
        )
        actual, milliseconds = _time_call(call, warmup, repeats)
        record["milliseconds"] = milliseconds
        record["output_shape"] = list(actual.shape)
        record["comparison"] = _compare_sample(actual, reference, rows, outputs)
        record["status"] = "pass"
    except Exception as error:  # Component probe must retain unsupported-layout evidence.
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _probe_stage(stage, experts, k_dim, n_dim, token_sizes, warmup, repeats):
    # This is the layout retained by the plugin after post-load conversion.
    weight_enk = torch.randint(
        -127,
        128,
        (experts, n_dim, k_dim),
        dtype=torch.int8,
        device="npu",
    )
    weight_scale = (
        torch.rand(experts, n_dim, dtype=torch.float32, device="npu") * 0.004
        + 0.0001
    ).to(torch.bfloat16)
    _synchronize()
    base_bytes = _storage_bytes(weight_enk)
    transpose_view = weight_enk.transpose(1, 2)

    stage_record = {
        "stage": stage,
        "plugin_weight_shape_enk": list(weight_enk.shape),
        "native_logical_shape_ekn": list(transpose_view.shape),
        "plugin_weight_storage_bytes": base_bytes,
        "transpose_view_shares_storage": (
            transpose_view.untyped_storage().data_ptr()
            == weight_enk.untyped_storage().data_ptr()
        ),
        "transpose_view_incremental_storage_bytes": 0,
        "empty_expert_policy": "only expert ids divisible by 3 receive rows",
        "tokens": [],
    }

    for tokens in token_sizes:
        counts = _counts_with_empty_experts(tokens, experts)
        cumulative = counts.cumsum(0).to(device="npu")
        counts_npu = counts.to(device="npu")
        source = torch.randn(tokens, k_dim, dtype=torch.bfloat16, device="npu")
        x_q, token_scale = torch.ops.npu.npu_dynamic_quant(source)
        del source
        rows = _sample_rows(tokens)
        reference, rows_npu, output_ids_npu = _reference_sample(
            x_q,
            weight_enk,
            weight_scale,
            token_scale,
            counts,
            rows,
            outputs=64,
        )

        token_record = {
            "tokens": tokens,
            "nonempty_experts": int((counts != 0).sum().item()),
            "empty_experts": int((counts == 0).sum().item()),
            "max_rows_per_expert": int(counts.max().item()),
            "input_quant_dtype": str(x_q.dtype),
            "per_token_scale_dtype": str(token_scale.dtype),
            "weight_scale_dtype": str(weight_scale.dtype),
            "layouts": [],
        }
        token_record["layouts"].append(
            _probe_layout(
                "nd_transpose_view_group_list_cumulative",
                transpose_view,
                x_q,
                weight_scale,
                token_scale,
                cumulative,
                0,
                reference,
                rows_npu,
                output_ids_npu,
                warmup,
                repeats,
            )
        )

        # The decode-side routing API returns counts.  Validate that empty
        # groups have the same numerical contract, even though this probe's
        # target integration is the eager prefill path.
        token_record["layouts"].append(
            _probe_layout(
                "nd_transpose_view_group_list_counts",
                transpose_view,
                x_q,
                weight_scale,
                token_scale,
                counts_npu,
                1,
                reference,
                rows_npu,
                output_ids_npu,
                warmup,
                repeats,
            )
        )

        started = time.perf_counter()
        contiguous = transpose_view.contiguous()
        _synchronize()
        contiguous_conversion_ms = (time.perf_counter() - started) * 1000
        contiguous_record = _probe_layout(
            "nd_contiguous_copy_group_list_cumulative",
            contiguous,
            x_q,
            weight_scale,
            token_scale,
            cumulative,
            0,
            reference,
            rows_npu,
            output_ids_npu,
            warmup,
            repeats,
        )
        contiguous_record["one_time_conversion_ms"] = contiguous_conversion_ms
        contiguous_record["incremental_storage_bytes"] = _storage_bytes(contiguous)
        token_record["layouts"].append(contiguous_record)
        del contiguous

        # npu_format_cast is disabled by default in this image.  Allocate a
        # genuine NZ tensor explicitly to measure the alternative without
        # retaining both copies across model layers.
        started = time.perf_counter()
        nz = torch_npu.empty_with_format(
            tuple(transpose_view.shape),
            dtype=transpose_view.dtype,
            device=transpose_view.device,
            acl_format=torch_npu.Format.FRACTAL_NZ,
        )
        nz.copy_(transpose_view)
        _synchronize()
        nz_conversion_ms = (time.perf_counter() - started) * 1000
        nz_record = _probe_layout(
            "fractal_nz_temporary_group_list_cumulative",
            nz,
            x_q,
            weight_scale,
            token_scale,
            cumulative,
            0,
            reference,
            rows_npu,
            output_ids_npu,
            warmup,
            repeats,
        )
        nz_record["one_time_conversion_ms"] = nz_conversion_ms
        nz_record["incremental_storage_bytes"] = _storage_bytes(nz)
        token_record["layouts"].append(nz_record)
        del nz, x_q, token_scale, reference
        torch.npu.empty_cache()
        stage_record["tokens"].append(token_record)

    del transpose_view, weight_enk, weight_scale
    torch.npu.empty_cache()
    return stage_record


def _tensor_difference(left, right):
    difference = (left.float() - right.float()).abs()
    reference_scale = right.float().abs().max().item()
    maximum = difference.max().item()
    return {
        "max_abs_error": maximum,
        "mean_abs_error": difference.mean().item(),
        "reference_max_abs": reference_scale,
        "normalized_max_error": maximum / max(reference_scale, 1e-12),
        "finite": bool(
            torch.isfinite(left).all().item() and torch.isfinite(right).all().item()
        ),
    }


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import probe dependency from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe_two_stage_routing(
    experts, routed_sizes, warmup, repeats, triton_source_dir
):
    """Compare native and current Triton MoE using real top-8 routing flow."""
    gmm = _load_source_module(
        "hy4_probe_triton_moe_gmm",
        triton_source_dir / "hy4_triton_moe_gmm.py",
    )
    projection = _load_source_module(
        "hy4_probe_triton_projection",
        triton_source_dir / "hy4_triton_projection.py",
    )
    clamp = projection.hy4_clamped_swiglu

    w13 = torch.randint(
        -127, 128, (experts, 128, 6144), dtype=torch.int8, device="npu"
    )
    s13 = (
        torch.rand(experts, 128, dtype=torch.float32, device="npu") * 0.004
        + 0.0001
    ).to(torch.bfloat16)
    w2 = torch.randint(
        -127, 128, (experts, 6144, 64), dtype=torch.int8, device="npu"
    )
    s2 = (
        torch.rand(experts, 6144, dtype=torch.float32, device="npu") * 0.004
        + 0.0001
    ).to(torch.bfloat16)
    active = torch.arange(0, experts, 3, dtype=torch.int32, device="npu")
    records = []

    for routed_tokens in routed_sizes:
        if routed_tokens % 8:
            raise ValueError("two-stage routed token count must be divisible by top_k=8")
        source_tokens = routed_tokens // 8
        hidden = torch.randn(
            source_tokens, 6144, dtype=torch.bfloat16, device="npu"
        )
        offsets = torch.arange(
            source_tokens * 8, dtype=torch.int64, device="npu"
        ).reshape(source_tokens, 8)
        topk_ids = active[offsets % active.numel()].to(torch.int32)
        topk_weights = torch.rand(
            source_tokens, 8, dtype=torch.float32, device="npu"
        )
        topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.bfloat16)
        row_idx = (
            torch.arange(
                source_tokens * 8, dtype=torch.int32, device="npu"
            )
            .view(8, -1)
            .permute(1, 0)
            .contiguous()
        )
        sorted_hidden, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=source_tokens,
            )
        )
        cumulative = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, experts
        ).to(torch.int64)
        counts = cumulative - torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device="npu"),
                cumulative[:-1],
            )
        )
        xq, xs = torch.ops.npu.npu_dynamic_quant(sorted_hidden)

        def native_core():
            native_g1 = _native_gmm(
                xq, w13.transpose(1, 2), s13, xs, cumulative, 0
            )
            native_act = clamp(native_g1)
            aq, activation_scale = torch.ops.npu.npu_dynamic_quant(native_act)
            native_g2 = _native_gmm(
                aq, w2.transpose(1, 2), s2, activation_scale, cumulative, 0
            )
            native_final = torch.ops.npu.npu_moe_finalize_routing(
                native_g2,
                skip1=None,
                skip2=None,
                bias=None,
                scales=topk_weights,
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=topk_ids,
            )
            return native_g1, native_act, native_g2, native_final

        def triton_core():
            x_dequant = xq.to(torch.bfloat16) * xs.to(torch.bfloat16).unsqueeze(1)
            triton_g1 = gmm.hy4_expert_gmm_i8(x_dequant, w13, s13, counts)
            triton_act = clamp(triton_g1)
            aq, activation_scale = torch.ops.npu.npu_dynamic_quant(triton_act)
            a_dequant = (
                aq.to(torch.bfloat16)
                * activation_scale.to(torch.bfloat16).unsqueeze(1)
            )
            triton_g2 = gmm.hy4_expert_gmm_i8(a_dequant, w2, s2, counts)
            triton_final = torch.ops.npu.npu_moe_finalize_routing(
                triton_g2,
                skip1=None,
                skip2=None,
                bias=None,
                scales=topk_weights,
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=topk_ids,
            )
            return triton_g1, triton_act, triton_g2, triton_final

        native_values, native_ms = _time_call(native_core, warmup, repeats)
        triton_values, triton_ms = _time_call(triton_core, warmup, repeats)
        labels = ("gmm1", "clamped_swiglu", "gmm2", "finalized")
        records.append(
            {
                "source_tokens": source_tokens,
                "routed_tokens": routed_tokens,
                "top_k": 8,
                "empty_experts": int((counts == 0).sum().item()),
                "native_milliseconds": native_ms,
                "triton_milliseconds": triton_ms,
                "triton_to_native_time_ratio": triton_ms / native_ms,
                "differences_native_vs_current_triton": {
                    label: _tensor_difference(native, triton)
                    for label, native, triton in zip(
                        labels, native_values, triton_values
                    )
                },
            }
        )
        torch.npu.empty_cache()

    torch.npu.empty_cache()
    return {
        "contract": (
            "npu_moe_init_routing top-8 -> A8 native GMM1 -> HYV4 clamp/SwiGLU "
            "-> fresh A8 native GMM2 -> npu_moe_finalize_routing"
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 1024, 4096])
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compare-triton-two-stage", action="store_true")
    parser.add_argument("--skip-layout-probe", action="store_true")
    parser.add_argument("--triton-source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.npu.set_device(0)
    torch.manual_seed(20260905)
    result = {
        "purpose": "HYV4 native NPU INT8 grouped-matmul prefill feasibility",
        "device": torch.npu.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "operator_schema": str(torch.ops.npu.npu_grouped_matmul._schemas),
        "stages": [] if args.skip_layout_probe else [
            _probe_stage(
                "gmm1_gate_up",
                args.experts,
                k_dim=6144,
                n_dim=128,
                token_sizes=args.tokens,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            _probe_stage(
                "gmm2_down",
                args.experts,
                k_dim=64,
                n_dim=6144,
                token_sizes=args.tokens,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
        ],
    }
    if args.compare_triton_two_stage:
        source_dir = args.triton_source_dir
        if source_dir is None:
            source_dir = (
                Path(__file__).resolve().parents[2]
                / "sglang_fl"
                / "models"
                / "hy4_preview"
            )
        result["two_stage_top8_comparison"] = _probe_two_stage_routing(
            args.experts,
            args.tokens,
            args.warmup,
            args.repeats,
            source_dir,
        )
    result["overall_pass"] = all(
        layout["status"] == "pass"
        and layout["comparison"]["allclose_atol_2e-2_rtol_3e-2"]
        for stage in result["stages"]
        for token in stage["tokens"]
        for layout in token["layouts"]
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
