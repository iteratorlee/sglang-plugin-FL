"""Focused HYV4 accuracy checks for active OOT operators on Ascend NPU.

The Layer-1 FlagGems whitelist contains only ``ones``.  RMSNorm,
SiluAndMul, and generic RotaryEmbedding are nevertheless available through
the plugin's independent Layer-2 OOT dispatch.  HYV4 overrides its own rotary
objects back to SGLang's native NPU method and uses a custom clamped SwiGLU for
MoE experts, so both the generic OOT and model-specific paths are checked.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  # Registers the torch.npu backend.
from safetensors import safe_open
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


MODEL_ROOT = Path("/models/Hy4-preview-W8A8-linear-moe")
EPSILON = 1e-5
ROPE_DIM = 64
ROPE_BASE = 10_000_000
MAX_POSITION = 1_048_576

NORM_WEIGHTS = {
    "hidden": "model.layers.0.input_layernorm.weight",
    "q_lora": "model.layers.0.self_attn.q_a_layernorm.weight",
    "kv_lora": "model.layers.0.self_attn.kv_a_layernorm.weight",
    "index_k": "model.layers.0.self_attn.indexer.k_norm.weight",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe_callable(fn):
    target = getattr(fn, "__func__", fn)
    return f"{target.__module__}.{target.__qualname__}"


def _load_checkpoint_tensor(name):
    index_path = MODEL_ROOT / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard = MODEL_ROOT / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    return tensor, shard


def _make_values(rows, width, seed, scale=1.0, extreme=False):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, width, generator=generator) * scale
    if extreme:
        flat = values.view(-1)
        probes = [100.0, -100.0, 32.0, -32.0, 10.0, -10.0, 1e-4, -1e-4]
        for index, value in enumerate(probes):
            flat[index] = value
        if width * rows > 65536:
            for index, value in enumerate(reversed(probes)):
                flat[65528 + index] = value
    return values.to(torch.bfloat16)


def _metrics(actual, expected):
    actual_fp32 = actual.float().cpu()
    expected_fp32 = expected.float().cpu()
    diff = actual_fp32 - expected_fp32
    reference_max = expected_fp32.abs().max().item()
    reference_l2 = torch.linalg.vector_norm(expected_fp32).item()
    max_abs = diff.abs().max().item()
    return {
        "max_abs": max_abs,
        "mean_abs": diff.abs().mean().item(),
        "relative_l2": torch.linalg.vector_norm(diff).item()
        / max(reference_l2, 1e-12),
        "max_abs_over_reference_max": max_abs / max(reference_max, 1e-12),
        "reference_max_abs": reference_max,
    }


def _metric_pass(metrics, normalized_limit=0.02, relative_l2_limit=0.01):
    return (
        math.isfinite(metrics["max_abs"])
        and metrics["max_abs_over_reference_max"] < normalized_limit
        and metrics["relative_l2"] < relative_l2_limit
    )


def _rms_reference(x, weight, residual=None, post_residual=None):
    summed = x.float()
    if residual is not None:
        summed = summed + residual.float()
    if post_residual is not None:
        summed = summed + post_residual.float()
    variance = summed.square().mean(-1, keepdim=True)
    output = summed * torch.rsqrt(variance + EPSILON) * weight.float()
    return output, summed


def _validate_rms_case(
    rms_cls,
    name,
    weight,
    tokens,
    seed,
    residual,
    post_residual,
    extreme,
):
    width = weight.numel()
    x_cpu = _make_values(tokens, width, seed, scale=0.75, extreme=extreme)
    residual_cpu = (
        _make_values(tokens, width, seed + 1, scale=0.5, extreme=extreme)
        if residual
        else None
    )
    post_cpu = (
        _make_values(tokens, width, seed + 2, scale=0.25, extreme=extreme)
        if post_residual
        else None
    )
    expected_output, expected_residual = _rms_reference(
        x_cpu, weight, residual_cpu, post_cpu
    )

    module = rms_cls(width, eps=EPSILON, weight_dtype=weight.dtype).npu()
    module.weight.data.copy_(weight.npu())
    arguments = (
        x_cpu.npu(),
        None if residual_cpu is None else residual_cpu.npu(),
        None if post_cpu is None else post_cpu.npu(),
    )
    native = module.forward_npu(*arguments)
    torch.npu.synchronize()
    if residual_cpu is None:
        native_output = native
        native_residual = None
    else:
        native_output, native_residual = native
    native_output_metrics = _metrics(native_output, expected_output)
    native_residual_metrics = (
        _metrics(native_residual, expected_residual)
        if native_residual is not None
        else None
    )

    try:
        oot_x = x_cpu.npu()
        oot_residual = None if residual_cpu is None else residual_cpu.npu()
        oot_post = None if post_cpu is None else post_cpu.npu()
        actual = module(
            oot_x,
            oot_residual,
            oot_post,
        )
        torch.npu.synchronize()
        error = None
    except Exception as exception:  # noqa: BLE001 - capture backend failures.
        actual = None
        error = {
            "type": type(exception).__name__,
            "message": str(exception)[:4096],
        }

    actual_output = None
    actual_residual = None
    if residual_cpu is None:
        actual_output = actual
    elif actual is not None:
        actual_output, actual_residual = actual

    output_metrics = (
        _metrics(actual_output, expected_output)
        if actual_output is not None
        else None
    )
    residual_metrics = (
        _metrics(actual_residual, expected_residual)
        if actual_residual is not None
        else None
    )
    result = {
        "name": name,
        "tokens": tokens,
        "width": width,
        "residual": residual,
        "post_residual": post_residual,
        "extreme": extreme,
        "forward_method": _describe_callable(module._forward_method),
        "oot": {
            "output": output_metrics,
            "returned_residual": residual_metrics,
            "finite": bool(
                actual_output is not None
                and torch.isfinite(actual_output).all().item()
            ),
            "error": error,
        },
        "native_npu": {
            "output": native_output_metrics,
            "returned_residual": native_residual_metrics,
            "finite": bool(torch.isfinite(native_output).all().item()),
        },
    }
    result["passed"] = (
        result["oot"]["finite"]
        and output_metrics is not None
        and _metric_pass(output_metrics)
        and (
            residual_metrics is None
            or _metric_pass(
                residual_metrics,
                normalized_limit=0.02,
                relative_l2_limit=0.01,
            )
        )
        and "rms_norm_bridge" in result["forward_method"]
    )
    result["native_npu"]["passed"] = (
        result["native_npu"]["finite"]
        and _metric_pass(native_output_metrics)
        and (
            native_residual_metrics is None
            or _metric_pass(native_residual_metrics)
        )
    )
    del module, actual, native
    return result


def _validate_rms(rms_cls):
    loaded = {}
    weight_metadata = {}
    for label, tensor_name in NORM_WEIGHTS.items():
        weight, shard = _load_checkpoint_tensor(tensor_name)
        loaded[label] = weight
        weight_metadata[label] = {
            "tensor": tensor_name,
            "shard": shard.name,
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "min": weight.float().min().item(),
            "max": weight.float().max().item(),
            "mean": weight.float().mean().item(),
        }

    specifications = [
        ("hidden_decode", "hidden", 1, False, False, False),
        ("hidden_prefill37", "hidden", 37, False, False, False),
        ("q_lora_prefill1024", "q_lora", 1024, False, False, False),
        ("kv_lora_prefill37_extreme", "kv_lora", 37, False, False, True),
        ("index_k_decode_extreme", "index_k", 1, False, False, True),
        ("hidden_decode_residual", "hidden", 1, True, False, False),
        ("hidden_prefill37_residual_post", "hidden", 37, True, True, False),
    ]
    cases = []
    for index, specification in enumerate(specifications):
        name, label, tokens, residual, post_residual, extreme = specification
        cases.append(
            _validate_rms_case(
                rms_cls,
                name,
                loaded[label],
                tokens,
                20260940 + index * 3,
                residual,
                post_residual,
                extreme,
            )
        )
        gc.collect()
        torch.npu.empty_cache()
    return {
        "checkpoint_weights": weight_metadata,
        "cases": cases,
        "active_hy4_non_residual_passed": all(
            case["passed"] for case in cases if not case["residual"]
        ),
        "native_npu_all_passed": all(
            case["native_npu"]["passed"] for case in cases
        ),
        "passed": all(case["passed"] for case in cases),
    }


def _activation_reference(x, clamped):
    width = x.shape[-1] // 2
    gate = x[..., :width].float()
    up = x[..., width:].float()
    if clamped:
        gate = gate.clamp(max=10.0)
        up = up.clamp(min=-10.0, max=10.0)
    return (F.silu(gate) * up).to(torch.bfloat16)


def _native_clamped_swiglu(x):
    """Eager prefill fallback using shape-safe native tensor operators."""
    width = x.shape[-1] // 2
    gate = x[..., :width].clamp(max=10.0)
    up = x[..., width:].clamp(min=-10.0, max=10.0)
    return F.silu(gate) * up


def _validate_checkpoint_gmm_activation(clamped_swiglu):
    """Exercise the clamp on native GMM1 output from a real W8A8 expert."""
    prefix = "model.layers.1.mlp.experts.0"
    tensor_names = {
        "gate_weight": f"{prefix}.gate_proj.weight",
        "up_weight": f"{prefix}.up_proj.weight",
        "gate_scale": f"{prefix}.gate_proj.weight_scale",
        "up_scale": f"{prefix}.up_proj.weight_scale",
    }
    tensors = {}
    shards = {}
    for label, tensor_name in tensor_names.items():
        tensor, shard = _load_checkpoint_tensor(tensor_name)
        tensors[label] = tensor
        shards[label] = shard.name

    local_width = 2048 // 32
    w13 = torch.cat(
        (
            tensors["gate_weight"][:local_width],
            tensors["up_weight"][:local_width],
        ),
        dim=0,
    ).unsqueeze(0)
    w13_scale = torch.cat(
        (
            tensors["gate_scale"][:local_width].flatten(),
            tensors["up_scale"][:local_width].flatten(),
        ),
        dim=0,
    ).to(torch.bfloat16).unsqueeze(0)
    norm_weight, norm_shard = _load_checkpoint_tensor(NORM_WEIGHTS["hidden"])
    w13_npu = w13.npu()
    w13_scale_npu = w13_scale.npu()
    cases = []
    for index, rows in enumerate((8, 296)):
        generator = torch.Generator().manual_seed(20260980 + index)
        # A standard-normal vector is already RMS-normalized. Multiplication by
        # the actual layer-0 RMS weight gives a representative model input
        # scale without loading the full transformer.
        hidden_cpu = (
            torch.randn(rows, 6144, generator=generator)
            * norm_weight.float()
        ).to(torch.bfloat16)
        hidden_npu = hidden_cpu.npu()
        hidden_q, hidden_scale = torch.ops.npu.npu_dynamic_quant(hidden_npu)
        cumulative = torch.tensor([rows], dtype=torch.int64, device="npu")
        g1 = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_q],
            weight=[w13_npu.transpose(1, 2)],
            scale=[w13_scale_npu],
            per_token_scale=[hidden_scale],
            group_list=cumulative,
            split_item=2,
            group_type=0,
            group_list_type=0,
            output_dtype=torch.bfloat16,
        )[0]
        torch.npu.synchronize()
        actual = clamped_swiglu(g1)
        native = _native_clamped_swiglu(g1)
        torch.npu.synchronize()
        g1_cpu = g1.float().cpu()
        expected = _activation_reference(g1.cpu(), clamped=True)
        actual_metrics = _metrics(actual, expected)
        native_metrics = _metrics(native, expected)
        gate = g1_cpu[:, :local_width]
        up = g1_cpu[:, local_width:]
        row = {
            "routed_rows": rows,
            "gmm1_shape": list(g1.shape),
            "gmm1_distribution": {
                "min": g1_cpu.min().item(),
                "max": g1_cpu.max().item(),
                "mean": g1_cpu.mean().item(),
                "std": g1_cpu.std().item(),
                "gate_above_limit_fraction": (gate > 10.0).float().mean().item(),
                "up_outside_limits_fraction": (
                    (up < -10.0) | (up > 10.0)
                ).float().mean().item(),
            },
            "production_activation_error": actual_metrics,
            "bf16_native_control_error": native_metrics,
            "finite": bool(
                torch.isfinite(g1).all().item()
                and torch.isfinite(actual).all().item()
            ),
        }
        row["passed"] = row["finite"] and _metric_pass(
            actual_metrics, normalized_limit=0.001, relative_l2_limit=0.001
        )
        cases.append(row)

    return {
        "checkpoint_tensors": tensor_names,
        "checkpoint_shards": shards,
        "rms_weight_shard": norm_shard.name,
        "tp_rank_slice": "expert-0 output channels [0:64] for TP rank 0",
        "w13_shape": list(w13.shape),
        "w13_scale_shape": list(w13_scale.shape),
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def _validate_activation_case(
    name,
    rows,
    width,
    seed,
    extreme,
    fn,
    clamped,
    forward_method,
    native_fn=None,
):
    x_cpu = _make_values(
        rows, 2 * width, seed, scale=2.0, extreme=extreme
    )
    expected = _activation_reference(x_cpu, clamped)
    try:
        oot_input = x_cpu.npu()
        actual = fn(oot_input)
        torch.npu.synchronize()
        metrics = _metrics(actual, expected)
        error = None
    except Exception as exception:  # noqa: BLE001 - capture backend failures.
        actual = None
        metrics = None
        error = {
            "type": type(exception).__name__,
            "message": str(exception)[:4096],
        }
    native = None
    native_metrics = None
    native_error = None
    if native_fn is not None:
        try:
            native_input = x_cpu.npu()
            native = native_fn(native_input)
            torch.npu.synchronize()
            native_metrics = _metrics(native, expected)
        except Exception as exception:  # noqa: BLE001
            native_error = {
                "type": type(exception).__name__,
                "message": str(exception)[:4096],
            }
    result = {
        "name": name,
        "input_shape": [rows, 2 * width],
        "output_shape": list(actual.shape) if actual is not None else None,
        "extreme": extreme,
        "clamped": clamped,
        "forward_method": forward_method,
        "error": metrics,
        "finite": bool(
            actual is not None and torch.isfinite(actual).all().item()
        ),
        "exception": error,
        "native_npu": {
            "error": native_metrics,
            "finite": bool(
                native is not None and torch.isfinite(native).all().item()
            ),
            "exception": native_error,
        }
        if native_fn is not None
        else None,
    }
    result["passed"] = (
        result["finite"]
        and result["output_shape"] == [rows, width]
        and metrics is not None
        and _metric_pass(metrics, normalized_limit=0.02, relative_l2_limit=0.01)
    )
    if result["native_npu"] is not None:
        result["native_npu"]["passed"] = (
            result["native_npu"]["finite"]
            and native_metrics is not None
            and _metric_pass(native_metrics)
        )
    del x_cpu, expected, actual, native
    return result


def _validate_activations(silu_cls, clamped_swiglu):
    silu = silu_cls()
    silu_method = _describe_callable(silu._forward_method)
    cases = [
        _validate_activation_case(
            "dense_layer0_decode_tp32",
            1,
            18432 // 32,
            20260960,
            False,
            silu,
            False,
            silu_method,
            silu.forward_npu,
        ),
        _validate_activation_case(
            "dense_layer0_prefill37_tp32_extreme",
            37,
            18432 // 32,
            20260961,
            True,
            silu,
            False,
            silu_method,
            silu.forward_npu,
        ),
        _validate_activation_case(
            "shared_expert_prefill1024_tp32",
            1024,
            2048 // 32,
            20260962,
            True,
            silu,
            False,
            silu_method,
            silu.forward_npu,
        ),
        _validate_activation_case(
            "routed_moe_decode_top8_tp32_clamped",
            8,
            2048 // 32,
            20260963,
            True,
            clamped_swiglu,
            True,
            "sglang_fl.models.hy4_preview.hy4_triton_projection.hy4_clamped_swiglu",
            _native_clamped_swiglu,
        ),
        _validate_activation_case(
            "routed_moe_prefill37_top8_tp32_clamped",
            37 * 8,
            2048 // 32,
            20260964,
            True,
            clamped_swiglu,
            True,
            "sglang_fl.models.hy4_preview.hy4_triton_projection.hy4_clamped_swiglu",
            _native_clamped_swiglu,
        ),
        _validate_activation_case(
            "routed_moe_prefill1024_top8_tp32_clamped",
            1024 * 8,
            2048 // 32,
            20260965,
            True,
            clamped_swiglu,
            True,
            "sglang_fl.models.hy4_preview.hy4_triton_projection.hy4_clamped_swiglu",
            _native_clamped_swiglu,
        ),
    ]
    checkpoint_gmm = _validate_checkpoint_gmm_activation(clamped_swiglu)
    return {
        "oot_forward_method": silu_method,
        "tp_size": 32,
        "dense_local_width": 18432 // 32,
        "moe_local_width": 2048 // 32,
        "swiglu_limit": 10.0,
        "cases": cases,
        "checkpoint_gmm1_activation": checkpoint_gmm,
        "passed": (
            "silu_and_mul_bridge" in silu_method
            and all(case["passed"] for case in cases)
            and checkpoint_gmm["passed"]
        ),
    }


def _rope_reference(query, key, rows):
    cos, sin = rows.float().cpu().chunk(2, dim=-1)

    def rotate(value):
        value = value.float().cpu()
        even = value[..., ::2]
        odd = value[..., 1::2]
        out_even = even * cos.unsqueeze(1) - odd * sin.unsqueeze(1)
        out_odd = odd * cos.unsqueeze(1) + even * sin.unsqueeze(1)
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    return rotate(query), rotate(key)


def _validate_rope_case(rope, name, tokens, start, seed, extreme):
    positions_cpu = torch.arange(start, start + tokens, dtype=torch.int64)
    query_cpu = _make_values(
        tokens, 2 * ROPE_DIM, seed, scale=1.0, extreme=extreme
    ).view(tokens, 2, ROPE_DIM)
    key_cpu = _make_values(
        tokens, ROPE_DIM, seed + 1, scale=1.0, extreme=extreme
    ).view(tokens, 1, ROPE_DIM)
    positions = positions_cpu.npu()
    rows = rope.cos_sin_cache.index_select(0, positions)
    expected = _rope_reference(query_cpu, key_cpu, rows)

    try:
        oot_query = query_cpu.npu()
        oot_key = key_cpu.npu()
        oot_output = rope(positions, oot_query, oot_key)
        torch.npu.synchronize()
        oot_errors = {
            "query": _metrics(oot_output[0], expected[0]),
            "key": _metrics(oot_output[1], expected[1]),
        }
        oot_error = None
    except Exception as exception:  # noqa: BLE001 - capture backend failures.
        oot_output = None
        oot_errors = None
        oot_error = {
            "type": type(exception).__name__,
            "message": str(exception)[:4096],
        }

    rope.sin_cos_cache = rows
    try:
        native_query = query_cpu.npu()
        native_key = key_cpu.npu()
        native_output = rope.forward_npu(positions, native_query, native_key)
        torch.npu.synchronize()
        native_errors = {
            "query": _metrics(native_output[0], expected[0]),
            "key": _metrics(native_output[1], expected[1]),
        }
        native_error = None
    except Exception as exception:  # noqa: BLE001
        native_output = None
        native_errors = None
        native_error = {
            "type": type(exception).__name__,
            "message": str(exception)[:4096],
        }

    result = {
        "name": name,
        "tokens": tokens,
        "positions": [start, start + tokens - 1],
        "extreme": extreme,
        "oot_error": oot_errors,
        "oot_exception": oot_error,
        "hy4_native_refreshed_error": native_errors,
        "hy4_native_exception": native_error,
        "finite": bool(
            oot_output is not None
            and native_output is not None
            and torch.isfinite(oot_output[0]).all().item()
            and torch.isfinite(oot_output[1]).all().item()
            and torch.isfinite(native_output[0]).all().item()
            and torch.isfinite(native_output[1]).all().item()
        ),
    }
    result["passed"] = (
        result["finite"]
        and oot_errors is not None
        and native_errors is not None
        and all(
            _metric_pass(
                metrics, normalized_limit=0.02, relative_l2_limit=0.01
            )
            for family in (oot_errors, native_errors)
            for metrics in family.values()
        )
    )
    return result


def _validate_rotary(rotary_cls):
    rope = rotary_cls(
        ROPE_DIM,
        ROPE_DIM,
        MAX_POSITION,
        ROPE_BASE,
        False,
        torch.bfloat16,
    ).npu()
    forward_method = _describe_callable(rope._forward_method)
    cases = [
        _validate_rope_case(rope, "decode", 1, 8193, 20260970, False),
        _validate_rope_case(
            rope, "prefill37_extreme", 37, 16384, 20260971, True
        ),
        _validate_rope_case(
            rope, "prefill1024", 1024, 32768, 20260972, False
        ),
    ]
    source = Path(inspect.getsourcefile(rotary_cls)).resolve()
    return {
        "generic_oot_forward_method": forward_method,
        "hy4_runtime_override": "RotaryEmbedding.forward_npu with layer-0 cache refresh",
        "source": str(source),
        "source_sha256": _sha256(source),
        "cases": cases,
        "passed": (
            "rotary_embedding_bridge" in forward_method
            and all(case["passed"] for case in cases)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path=str(MODEL_ROOT),
            device="npu",
            dtype="bfloat16",
            attention_backend="ascend",
        )
    )

    import sglang_fl

    sglang_fl.load_plugin()
    HookRegistry.apply_hooks()

    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang_fl.dispatch import resolve_op
    from sglang_fl.models.hy4_preview.hy4_triton_projection import (
        hy4_clamped_swiglu,
    )

    resolved = {
        name: _describe_callable(resolve_op(name))
        for name in ("rms_norm", "silu_and_mul", "rotary_embedding")
    }
    result = {
        "metadata": {
            "physical_device_from_env": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "sglang_version": importlib.metadata.version("sglang"),
            "flag_gems_version": importlib.metadata.version("flag_gems"),
            "sglang_fl_file": sglang_fl.__file__,
            "plugin_active": sglang_fl.is_plugin_active(),
            "use_flaggems": os.environ.get("USE_FLAGGEMS"),
            "layer1_flaggems_whitelist": os.environ.get(
                "SGLANG_FL_FLAGOS_WHITELIST"
            ),
            "layer2_preference": os.environ.get("SGLANG_FL_PREFER", "flagos"),
            "resolved_layer2_implementations": resolved,
        },
        "rms_norm": _validate_rms(RMSNorm),
        "activations": _validate_activations(SiluAndMul, hy4_clamped_swiglu),
        "rotary": _validate_rotary(RotaryEmbedding),
    }
    result["passed"] = (
        result["metadata"]["plugin_active"]
        and all("flagos" in value for value in resolved.values())
        and result["rms_norm"]["passed"]
        and result["activations"]["passed"]
        and result["rotary"]["passed"]
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
