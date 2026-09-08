"""Validate HYV4's shared RoPE cache ordering on Ascend NPU.

SGLang 0.5.11's RoPE factory returns one object for identical arguments.  The
NPU forward path then consumes ``sin_cos_cache`` without indexing ``positions``
when that attribute exists.  This probe reproduces the resulting cross-chunk
stale-cache error and validates the layer-0 refresh pattern in eager prefill
and graph decode without loading a full model.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa
from sglang.srt.layers.rotary_embedding import factory
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


HEAD_SIZE = 64
ROTARY_DIM = 64
MAX_POSITION = 1_048_576
ROPE_BASE = 10_000_000
IS_NEOX_STYLE = False
MAIN_HEADS = 2
INDEX_HEADS = 32


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_inputs(tokens, heads, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(
        tokens, heads, HEAD_SIZE, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    key = torch.randn(
        tokens, 1, HEAD_SIZE, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    return query.npu(), key.npu()


def _cpu_reference(query, key, cache_rows):
    cos, sin = cache_rows.float().cpu().chunk(2, dim=-1)

    def rotate(value):
        value = value.float().cpu()
        even = value[..., ::2]
        odd = value[..., 1::2]
        out_even = even * cos.unsqueeze(1) - odd * sin.unsqueeze(1)
        out_odd = odd * cos.unsqueeze(1) + even * sin.unsqueeze(1)
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    return rotate(query), rotate(key)


def _max_abs(actual, expected):
    return (actual.float().cpu() - expected.float().cpu()).abs().max().item()


def _error_pair(actual, expected):
    return {
        "query": _max_abs(actual[0], expected[0]),
        "key": _max_abs(actual[1], expected[1]),
    }


def _max_pair(errors):
    return max(errors["query"], errors["key"])


def _new_rope():
    return factory.get_rope_wrapper(
        HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position=MAX_POSITION,
        base=ROPE_BASE,
        rope_scaling=None,
        is_neox_style=IS_NEOX_STYLE,
        dtype=torch.bfloat16,
        device="npu",
    )


def _forward_with_rows(rope, positions, query, key, rows):
    rope.sin_cos_cache = rows
    return rope.forward_npu(positions, query, key)


def _validate_identity_and_stale(main_rope, indexer_rope):
    tokens = 1024
    previous_positions = torch.arange(tokens, dtype=torch.int64, device="npu")
    next_positions = torch.arange(
        tokens, 2 * tokens, dtype=torch.int64, device="npu"
    )
    previous_rows = main_rope.cos_sin_cache.index_select(0, previous_positions)
    next_rows = main_rope.cos_sin_cache.index_select(0, next_positions)

    # Match the stock layer-0 Indexer: it refreshes the shared object's cache
    # for the previous chunk, then invokes RoPE using that cached selection.
    index_query, index_key = _make_inputs(tokens, INDEX_HEADS, 20260911)
    indexer_output = _forward_with_rows(
        indexer_rope,
        previous_positions,
        index_query,
        index_key,
        previous_rows,
    )
    torch.npu.synchronize()

    main_query, main_key = _make_inputs(tokens, MAIN_HEADS, 20260912)
    stale_output = main_rope.forward_npu(
        next_positions, main_query.clone(), main_key.clone()
    )
    torch.npu.synchronize()
    stale_old_reference = _cpu_reference(main_query, main_key, previous_rows)
    correct_reference = _cpu_reference(main_query, main_key, next_rows)

    refreshed_output = _forward_with_rows(
        main_rope,
        next_positions,
        main_query.clone(),
        main_key.clone(),
        next_rows,
    )
    torch.npu.synchronize()

    stale_matches_previous = _error_pair(stale_output, stale_old_reference)
    stale_vs_correct = _error_pair(stale_output, correct_reference)
    refreshed_vs_correct = _error_pair(refreshed_output, correct_reference)
    result = {
        "factory_object_identity": main_rope is indexer_rope,
        "previous_positions": [0, tokens - 1],
        "next_positions": [tokens, 2 * tokens - 1],
        "indexer_output_finite": bool(
            torch.isfinite(indexer_output[0]).all().item()
            and torch.isfinite(indexer_output[1]).all().item()
        ),
        "stale_matches_previous_cache_max_abs": stale_matches_previous,
        "stale_vs_correct_positions_max_abs": stale_vs_correct,
        "refreshed_vs_correct_positions_max_abs": refreshed_vs_correct,
    }
    result["passed"] = (
        result["factory_object_identity"]
        and result["indexer_output_finite"]
        and _max_pair(stale_matches_previous) < 0.02
        and _max_pair(stale_vs_correct) > 0.05
        and _max_pair(refreshed_vs_correct) < 0.02
    )
    return result


def _validate_prefill(rope, tokens, start_position, seed):
    positions = torch.arange(
        start_position,
        start_position + tokens,
        dtype=torch.int64,
        device="npu",
    )
    query, key = _make_inputs(tokens, MAIN_HEADS, seed)
    rows = rope.cos_sin_cache.index_select(0, positions)
    expected = _cpu_reference(query, key, rows)
    actual = _forward_with_rows(rope, positions, query, key, rows)
    torch.npu.synchronize()
    errors = _error_pair(actual, expected)
    result = {
        "tokens": tokens,
        "positions": [start_position, start_position + tokens - 1],
        "max_abs_error": errors,
        "finite": bool(
            torch.isfinite(actual[0]).all().item()
            and torch.isfinite(actual[1]).all().item()
        ),
    }
    result["passed"] = result["finite"] and _max_pair(errors) < 0.02
    return result


def _validate_decode_graph(rope):
    positions_to_test = [4096, 8193, 16384, 131071, 524287]
    static_positions = torch.empty(1, dtype=torch.int64, device="npu")
    static_query = torch.empty(
        1, MAIN_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device="npu"
    )
    static_key = torch.empty(
        1, 1, HEAD_SIZE, dtype=torch.bfloat16, device="npu"
    )
    first_query, first_key = _make_inputs(1, MAIN_HEADS, 20260930)
    static_positions.fill_(positions_to_test[0])
    static_query.copy_(first_query)
    static_key.copy_(first_key)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        # This is the proposed layer-0 refresh: index_select is captured and
        # consumes the replay-updated positions tensor before native RoPE.
        rope.sin_cos_cache = rope.cos_sin_cache.index_select(0, static_positions)
        captured_output = rope.forward_npu(
            static_positions, static_query, static_key
        )

    rows = []
    previous_output = None
    for step, position in enumerate(positions_to_test):
        query, key = _make_inputs(1, MAIN_HEADS, 20260930 + step)
        static_positions.fill_(position)
        static_query.copy_(query)
        static_key.copy_(key)
        correct_rows = rope.cos_sin_cache.index_select(0, static_positions)
        eager_output = fused_rope_qk_mqa(
            static_query,
            static_key,
            correct_rows,
            ROTARY_DIM,
            IS_NEOX_STYLE,
        )
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()

        errors = _error_pair(captured_output, eager_output)
        output_cpu = captured_output[0].float().cpu().clone()
        changed = previous_output is None or not torch.equal(
            output_cpu, previous_output
        )
        row = {
            "step": step,
            "position": position,
            "graph_vs_eager_max_abs": errors,
            "output_changed": changed,
            "finite": bool(
                torch.isfinite(captured_output[0]).all().item()
                and torch.isfinite(captured_output[1]).all().item()
            ),
        }
        row["passed"] = (
            row["finite"] and row["output_changed"] and _max_pair(errors) < 0.02
        )
        rows.append(row)
        previous_output = output_cpu

    return {
        "capture_tokens": 1,
        "changed_position_replays": len(positions_to_test) - 1,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)
    torch.manual_seed(20260911)
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="/models/Hy4-preview-W8A8-linear-moe",
            device="npu",
        )
    )
    factory._ROPE_DICT.clear()
    main_rope = _new_rope()
    indexer_rope = _new_rope()
    main_rope.npu()

    source_path = Path(inspect.getsourcefile(main_rope.__class__)).resolve()
    result = {
        "metadata": {
            "physical_device_from_env": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "sglang_version": importlib.metadata.version("sglang"),
            "torch_npu_file": torch_npu.__file__,
            "rope_source": str(source_path),
            "rope_source_sha256": _sha256(source_path),
            "head_size": HEAD_SIZE,
            "rotary_dim": ROTARY_DIM,
            "max_position": MAX_POSITION,
            "base": ROPE_BASE,
            "is_neox_style": IS_NEOX_STYLE,
            "dtype": str(main_rope.cos_sin_cache.dtype),
            "cache_device": str(main_rope.cos_sin_cache.device),
        },
        "identity_and_stale_reproduction": _validate_identity_and_stale(
            main_rope, indexer_rope
        ),
        "refreshed_prefill": [
            _validate_prefill(main_rope, 37, 8192, 20260920),
            _validate_prefill(main_rope, 1024, 16384, 20260921),
        ],
        "refreshed_decode_graph": _validate_decode_graph(main_rope),
    }
    result["passed"] = (
        result["identity_and_stale_reproduction"]["passed"]
        and all(case["passed"] for case in result["refreshed_prefill"])
        and result["refreshed_decode_graph"]["passed"]
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
