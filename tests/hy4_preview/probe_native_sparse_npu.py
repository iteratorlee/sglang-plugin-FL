"""Inspect Ascend sparse MLA output/statistics and sink correction."""

import argparse
import json
import time

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--layout", default="TND")
    parser.add_argument("--sparse-mode", type=int, default=3)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20260905)
    t, length, h, dn, dr, page = args.tokens, args.context, 2, 512, 64, 128
    blocks = (length + page - 1) // page
    key = (torch.randn(blocks, page, 1, dn) * 0.2).bfloat16().npu()
    rope = (torch.randn(blocks, page, 1, dr) * 0.2).bfloat16().npu()
    query = (torch.randn(t, h, dn) * 0.2).bfloat16().npu()
    query_rope = (torch.randn(t, h, dr) * 0.2).bfloat16().npu()
    indices = torch.full((t, 1, 2048), -1, dtype=torch.int32)
    for row in range(t):
        end = length - t + row + 1
        ids = torch.randperm(end)[: min(end, 2048)]
        indices[row, 0, : ids.numel()] = ids
    indices = indices.npu()
    block_table = torch.arange(blocks, dtype=torch.int32).view(1, -1).npu()
    cu_q = torch.tensor([t], dtype=torch.int32, device="npu")
    lengths = torch.tensor([length], dtype=torch.int32, device="npu")

    def run():
        return torch_npu.npu_sparse_flash_attention(
            query=query,
            key=key if args.layout == "PA_BSND" else key.view(-1, 1, dn),
            value=key if args.layout == "PA_BSND" else key.view(-1, 1, dn),
            query_rope=query_rope,
            key_rope=rope if args.layout == "PA_BSND" else rope.view(-1, 1, dr),
            sparse_indices=indices,
            block_table=block_table if args.layout == "PA_BSND" else None,
            actual_seq_lengths_query=cu_q,
            actual_seq_lengths_kv=lengths,
            scale_value=256**-0.5,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv=args.layout,
            sparse_mode=args.sparse_mode,
            attention_mode=2,
            return_softmax_lse=True,
        )

    started = time.monotonic()
    outputs = run()
    torch.npu.synchronize()
    print("EAGER", time.monotonic() - started, flush=True)
    for index, value in enumerate(outputs):
        print(index, value.shape, value.dtype, value.flatten()[:16].cpu(), flush=True)
    ref = []
    ref_lse = []
    for row in range(t):
        ids = indices[row, 0].long()
        ids = ids[ids >= 0]
        k = key.view(-1, dn)[ids].float()
        kr = rope.view(-1, dr)[ids].float()
        logits = (query[row].float() @ k.T + query_rope[row].float() @ kr.T) / 16
        ref.append(logits.softmax(-1) @ k)
        ref_lse.append(logits.logsumexp(-1))
    expected = torch.stack(ref)
    lse = torch.stack(ref_lse)
    error = (outputs[0].float().reshape_as(expected) - expected).abs().max().item()
    print(
        "REFERENCE",
        json.dumps(
            {
                "max_abs_error": error,
                "finite": bool(torch.isfinite(outputs[0]).all()),
                "lse": lse.flatten()[:16].cpu().tolist(),
            }
        ),
        flush=True,
    )
    if args.graph:
        for _ in range(2):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = run()
        graph.replay()
        torch.npu.synchronize()
        print(
            "GRAPH",
            [
                (a.float() - b.float()).abs().max().item() if a.numel() else 0
                for a, b in zip(captured, outputs)
            ],
            flush=True,
        )
        indices.copy_(torch.roll(indices, 37, -1))
        query.mul_(0.5)
        eager_changed = run()
        graph.replay()
        torch.npu.synchronize()
        print(
            "CHANGED_GRAPH",
            [
                (a.float() - b.float()).abs().max().item() if a.numel() else 0
                for a, b in zip(captured, eager_changed)
            ],
            flush=True,
        )


if __name__ == "__main__":
    main()
