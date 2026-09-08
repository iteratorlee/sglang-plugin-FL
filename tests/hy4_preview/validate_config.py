#!/usr/bin/env python3
"""Validate HYV4 native-DSA context and NPUGraph deployment bounds."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from sglang_fl.models.hy4_preview.bootstrap import (
    HYV4Config,
    HYV4_SAFE_GRAPH_BATCH_SIZE,
    apply_sglang_patches,
)


LONG_CONTEXT_LENGTH = 32_768


def _model_config(config: HYV4Config) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        is_draft_model=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    from sglang.srt import server_args as server_args_module
    from sglang.srt.configs.model_config import ModelConfig

    apply_sglang_patches()
    config = HYV4Config(
        architectures=["HYV4ForCausalLM"],
        index_topk=2048,
        max_position_embeddings=1_048_576,
        indexer_types=["full", "shared"],
    )

    safe_args = SimpleNamespace(
        disable_cuda_graph=False,
        cuda_graph_bs=[1, HYV4_SAFE_GRAPH_BATCH_SIZE],
        cuda_graph_max_bs=HYV4_SAFE_GRAPH_BATCH_SIZE,
        max_running_requests=HYV4_SAFE_GRAPH_BATCH_SIZE,
    )
    server_args_module.set_global_server_args_for_scheduler(safe_args)
    long_context = _model_config(config)
    ModelConfig._derive_context_length(long_context, LONG_CONTEXT_LENGTH)

    unsafe_args = SimpleNamespace(
        disable_cuda_graph=False,
        cuda_graph_bs=[1, HYV4_SAFE_GRAPH_BATCH_SIZE + 1],
        cuda_graph_max_bs=HYV4_SAFE_GRAPH_BATCH_SIZE + 1,
        max_running_requests=HYV4_SAFE_GRAPH_BATCH_SIZE,
    )
    server_args_module.set_global_server_args_for_scheduler(unsafe_args)
    graph_error = ""
    try:
        ModelConfig._derive_context_length(
            _model_config(config), LONG_CONTEXT_LENGTH
        )
    except ValueError as exc:
        graph_error = str(exc)

    concurrency_errors = []
    for unsafe_max_running in (None, HYV4_SAFE_GRAPH_BATCH_SIZE + 1):
        unsafe_concurrency_args = SimpleNamespace(
            disable_cuda_graph=False,
            cuda_graph_bs=[1, HYV4_SAFE_GRAPH_BATCH_SIZE],
            cuda_graph_max_bs=HYV4_SAFE_GRAPH_BATCH_SIZE,
            max_running_requests=unsafe_max_running,
        )
        server_args_module.set_global_server_args_for_scheduler(
            unsafe_concurrency_args
        )
        try:
            ModelConfig._derive_context_length(
                _model_config(config), LONG_CONTEXT_LENGTH
            )
        except ValueError as exc:
            concurrency_errors.append(str(exc))

    server_args_module.set_global_server_args_for_scheduler(safe_args)
    accepted = _model_config(config)
    ModelConfig._derive_context_length(accepted, LONG_CONTEXT_LENGTH)

    result = {
        "checkpoint_context_limit": config.max_position_embeddings,
        "validated_request_context": LONG_CONTEXT_LENGTH,
        "native_dsa_enabled": config.index_topk == 2048,
        "graph_batch_limit": HYV4_SAFE_GRAPH_BATCH_SIZE,
        "long_context_accepted": long_context.context_len == LONG_CONTEXT_LENGTH,
        "graph_batch_rejected": "NPUGraph is validated only" in graph_error,
        "unsafe_concurrency_rejected": (
            len(concurrency_errors) == 2
            and all(
                "max_running_requests" in error
                for error in concurrency_errors
            )
        ),
        "safe_boundary_accepted": (
            accepted.context_len == LONG_CONTEXT_LENGTH
        ),
        "graph_error": graph_error,
        "concurrency_errors": concurrency_errors,
    }
    result["passed"] = all(
        result[key]
        for key in (
            "native_dsa_enabled",
            "long_context_accepted",
            "graph_batch_rejected",
            "unsafe_concurrency_rejected",
            "safe_boundary_accepted",
        )
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
