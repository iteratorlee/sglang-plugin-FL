#!/usr/bin/env python3
"""Validate HYV4 checkpoint vocabulary sharding and TP32 embedding reduction."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # Registers the torch.npu runtime.
from safetensors import safe_open

from sglang.srt.distributed.parallel_state import (
    destroy_model_parallel,
    get_tensor_model_parallel_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding


VOCAB_SIZE = 120832
HIDDEN_SIZE = 6144
TOKEN_IDS = (
    100,
    70000,
    119995,
    120000,
    120001,
    120025,
    120029,
    120030,
    120039,
)
WEIGHT_FILE = "model-00048-of-00131.safetensors"
WEIGHT_NAME = "model.embed_tokens.weight"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="/models/Hy4-preview-W8A8-linear-moe"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 32:
        raise RuntimeError(f"HYV4 vocab validation requires TP32, got {world_size}")

    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    embedding = VocabParallelEmbedding(
        VOCAB_SIZE,
        HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
    ).npu()
    indices = embedding.shard_indices
    start = indices.org_vocab_start_index
    end = indices.org_vocab_end_index
    weight_path = os.path.join(args.model_path, WEIGHT_FILE)
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        weight_slice = handle.get_slice(WEIGHT_NAME)
        local_weight = weight_slice[start:end, :]
        reference = torch.stack(
            [weight_slice[token_id, :] for token_id in TOKEN_IDS]
        )
    embedding.weight.data.copy_(local_weight)

    token_tensor = torch.tensor(TOKEN_IDS, dtype=torch.int64, device="npu")
    actual = embedding(token_tensor)
    torch.npu.synchronize()
    reference_npu = reference.to(device="npu", dtype=torch.bfloat16)
    row_errors = (
        actual.float() - reference_npu.float()
    ).abs().amax(dim=1)
    finite = bool(torch.isfinite(actual).all().item())
    local_copy_error = float(
        (embedding.weight.float() - local_weight.to("npu").float())
        .abs()
        .max()
        .item()
    )

    all_rank_ok = torch.tensor(
        int(finite and local_copy_error == 0.0 and row_errors.max().item() == 0.0),
        dtype=torch.int32,
        device="npu",
    )
    dist.all_reduce(
        all_rank_ok,
        op=dist.ReduceOp.MIN,
        group=get_tensor_model_parallel_group().device_group,
    )
    result = {
        "world_size": world_size,
        "vocab_size": VOCAB_SIZE,
        "partition_size": embedding.num_embeddings_per_partition,
        "rank0_range": [0, embedding.num_embeddings_per_partition],
        "rank31_range": [
            31 * embedding.num_embeddings_per_partition,
            32 * embedding.num_embeddings_per_partition,
        ],
        "token_ids": list(TOKEN_IDS),
        "row_max_abs_errors": [float(value) for value in row_errors.cpu()],
        "finite": finite,
        "local_checkpoint_copy_max_abs_error": local_copy_error,
        "all_ranks_passed": bool(all_rank_ok.item()),
        "passed": bool(all_rank_ok.item()),
    }
    if rank == 0:
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as output:
                output.write(rendered + "\n")

    dist.barrier()
    destroy_model_parallel()
    dist.destroy_process_group()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
