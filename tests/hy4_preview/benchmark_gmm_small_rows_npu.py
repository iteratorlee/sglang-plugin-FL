"""Microbenchmark graph GMM scheduling; not an end-to-end speedup claim."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch_npu


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--baseline-module", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    modules = {"baseline_expert_grid": load("old_gmm_bench", args.baseline_module),
               "small_row_grid": load("new_gmm_bench", args.module)}
    torch.manual_seed(20260905)
    stages = []
    for k_dim, n_dim in ((6144, 128), (64, 6144)):
        q = torch.randint(-127, 128, (8, k_dim), dtype=torch.int8, device="npu")
        scales = torch.rand(8, device="npu") * .01
        w = torch.randint(-127, 128, (256, n_dim, k_dim), dtype=torch.int8, device="npu")
        ws = (torch.rand(256, n_dim, device="npu") * .01).bfloat16()
        counts = torch.zeros(256, dtype=torch.int64, device="npu")
        counts[:8] = 1
        record = {"shape": [256, n_dim, k_dim], "routed_tokens": 8, "timings": {}}
        outputs = {}
        for name, module in modules.items():
            def call():
                return module.hy4_expert_gmm_i8(q, w, ws, counts, pertoken_scale=scales)
            for _ in range(2):
                call()
            torch.npu.synchronize()
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph):
                output = call()
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(args.repeats):
                graph.replay()
            torch.npu.synchronize()
            record["timings"][name] = (time.perf_counter() - start) * 1000 / args.repeats
            outputs[name] = output.cpu()
        record["output_exact"] = torch.equal(*outputs.values())
        record["speedup"] = record["timings"]["baseline_expert_grid"] / record["timings"]["small_row_grid"]
        stages.append(record)
        print(json.dumps(record), flush=True)
    result = {"repeats": args.repeats, "stages": stages,
              "passed": all(stage["output_exact"] for stage in stages)}
    Path(args.output).write_text(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
