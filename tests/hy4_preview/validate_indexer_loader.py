#!/usr/bin/env python3
"""Validate HYV4 full/shared indexer topology and rope-first permutation."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from sglang_fl.models.hy4_preview.hy4 import (
    _hyv4_expected_indexer_weights,
    _hyv4_full_indexer_layers,
    permute_hyv4_indexer_weight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    config = SimpleNamespace(
        num_hidden_layers=6,
        indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
        index_n_heads=2,
        index_head_dim=4,
        qk_rope_head_dim=1,
    )
    wq = torch.arange(8 * 3).reshape(8, 3)
    wk = torch.arange(4 * 3).reshape(4, 3)
    norm = torch.arange(4)
    unchanged = torch.arange(7)

    wq_result = permute_hyv4_indexer_weight(
        "model.layers.0.self_attn.indexer.wq_b.weight", wq, config
    )
    wk_result = permute_hyv4_indexer_weight(
        "model.layers.0.self_attn.indexer.wk.weight", wk, config
    )
    norm_result = permute_hyv4_indexer_weight(
        "model.layers.0.self_attn.indexer.k_norm.bias", norm, config
    )
    unchanged_result = permute_hyv4_indexer_weight(
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        unchanged,
        config,
    )
    expected_wq_rows = [3, 0, 1, 2, 7, 4, 5, 6]
    result = {
        "full_layers": _hyv4_full_indexer_layers(config),
        "expected_weight_count": len(_hyv4_expected_indexer_weights(config)),
        "wq_group_permutation": torch.equal(
            wq_result, wq[expected_wq_rows]
        ),
        "wk_group_permutation": torch.equal(wk_result, wk[[3, 0, 1, 2]]),
        "norm_group_permutation": torch.equal(
            norm_result, norm[[3, 0, 1, 2]]
        ),
        "weights_proj_unchanged": unchanged_result is unchanged,
    }
    result["passed"] = (
        result["full_layers"] == [0, 3]
        and result["expected_weight_count"] == 10
        and all(
            result[key]
            for key in (
                "wq_group_permutation",
                "wk_group_permutation",
                "norm_group_permutation",
                "weights_proj_unchanged",
            )
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
