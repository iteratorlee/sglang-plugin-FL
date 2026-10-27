"""Standalone torchrun SGGroup/FlagCX eager + NPUGraph acceptance probe.

Run only on the parent's isolated NPU candidate, never on a production service.
Requires explicit rank groups and a separate proof directory. No model is loaded.
No hardware modules are imported until main(), so CPU tests can check the oracle.
"""

import argparse
import hashlib
import json
import os
import socket
from collections import Counter
from datetime import timedelta
from pathlib import Path


def validate_groups(groups, world_size):
    if set(groups) != {"tp", "attention_tp"}:
        raise ValueError(
            "Supply exact tp AND attention_tp partitions from the candidate role"
        )
    for name, partition in groups.items():
        flat = [rank for ranks in partition for rank in ranks]
        if sorted(flat) != list(range(world_size)) or any(
            len(ranks) < 2 or ranks != sorted(ranks) for ranks in partition
        ):
            raise ValueError(
                f"{name} must partition all ranks exactly once into sorted groups of size >=2"
            )
    return groups


def rank_inputs(torch, global_rank, group_size, step):
    """Small exact FP32 integers encode global source rank, step and destination."""
    base = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    dense = base + 16 * (global_rank + 1) + 256 * step
    scatter = torch.cat([dense + 2 * dst for dst in range(group_size)], dim=0)
    return dense, scatter


def expected_outputs(torch, ranks, rank_in_group, step):
    inputs = [rank_inputs(torch, rank, len(ranks), step)[0] for rank in ranks]
    reduced = torch.stack(inputs).sum(dim=0)
    return {
        "group_all_reduce": reduced,
        "npu_all_reduce": reduced,
        "group_all_gather": torch.cat(inputs, dim=0),
        "npu_all_gather": torch.cat(inputs, dim=1),
        "group_reduce_scatter": reduced + len(ranks) * 2 * rank_in_group,
    }


def collectives(torch, group, dense, scatter):
    ar, nar = dense.clone(), dense.clone()
    if group.all_reduce(ar) is not ar:
        raise AssertionError("SGGroup all_reduce lost in-place identity")
    if group.npu_communicator.all_reduce(nar) is not nar:
        raise AssertionError("NpuCommunicator all_reduce lost in-place identity")
    gathered = torch.empty(
        (group.world_size * 2, 4), dtype=dense.dtype, device=dense.device
    )
    group.all_gather_into_tensor(gathered, dense)
    npu_gathered = group.npu_communicator.all_gather(dense, dim=-1)
    reduced = torch.empty_like(dense)
    group.reduce_scatter_tensor(reduced, scatter)
    return dict(
        group_all_reduce=ar,
        npu_all_reduce=nar,
        group_all_gather=gathered,
        npu_all_gather=npu_gathered,
        group_reduce_scatter=reduced,
    )


def assert_outputs(torch, group, outputs, step):
    torch.npu.synchronize()
    expected = expected_outputs(torch, group.ranks, group.rank_in_group, step)
    for name, output in outputs.items():
        torch.testing.assert_close(
            output.cpu(),
            expected[name],
            rtol=0,
            atol=0,
            msg=f"{group.unique_name} step={step} {name}",
        )


def run_group(torch, dist, group, mode, emit):
    comm = group._fl_native_comm
    if comm is None or comm.disabled or not comm.available:
        raise RuntimeError("Native FlagCX transport was not installed")
    submissions = Counter()
    observer = comm._collective_observer

    def observe(api, tensor):
        submissions[api] += 1
        observer(api, tensor)

    comm._collective_observer = observe
    device = group.device
    dense, scatter = [
        value.to(device)
        for value in rank_inputs(torch, dist.get_rank(), group.world_size, 0)
    ]
    for step in (0, 1):
        values = rank_inputs(torch, dist.get_rank(), group.world_size, step)
        dense.copy_(values[0])
        scatter.copy_(values[1])
        outputs = collectives(torch, group, dense, scatter)
        assert_outputs(torch, group, outputs, step)
    required = {"flagcxAllReduce": 4, "flagcxAllGather": 4, "flagcxReduceScatter": 2}
    if dict(submissions) != required:
        raise AssertionError(f"Eager CAPI submission mismatch: {submissions}")
    emit(
        "eager_verified",
        group=group.unique_name,
        ranks=group.ranks,
        submissions=dict(submissions),
        rank_in_group=group.rank_in_group,
    )
    if mode == "eager":
        return

    # Warm the SAME stream used for capture, including FlagCX streamCopy cache.
    stream = torch.npu.Stream()
    with group.graph_capture(stream=stream):
        outputs = collectives(torch, group, dense, scatter)
    assert_outputs(torch, group, outputs, 1)
    dist.barrier(group=group.cpu_group)
    graph = torch.npu.NPUGraph()
    with group.graph_capture(stream=stream):
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            outputs = collectives(torch, group, dense, scatter)
    before_replay = dict(submissions)
    for step in (2, 3, 4):
        values = rank_inputs(torch, dist.get_rank(), group.world_size, step)
        dense.copy_(values[0])
        scatter.copy_(values[1])
        # Input copies complete before replay; every rank follows the same sequence.
        torch.npu.synchronize()
        dist.barrier(group=group.cpu_group)
        graph.replay()
        assert_outputs(torch, group, outputs, step)
        if dict(submissions) != before_replay:
            raise AssertionError(
                "Graph replay unexpectedly re-entered Python collective calls"
            )
        emit(
            "graph_replay_verified",
            group=group.unique_name,
            step=step,
            ranks=group.ranks,
            rank_in_group=group.rank_in_group,
            capture_submissions=before_replay,
            graph_api="torch.npu.graph(auto_dispatch_capture=True)",
        )
    dist.barrier(group=group.cpu_group)
    # Graph object/buffers are released before communicator teardown.
    del graph, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-file", type=Path, required=True)
    parser.add_argument("--proof-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "both"), default="both")
    args = parser.parse_args()
    if os.environ.get("SGLANG_FL_INTEGRATION") != "native":
        parser.error("Set SGLANG_FL_INTEGRATION=native before starting torchrun")
    # Namespace all CAPI records from this process as smoke, never serving proof.
    args.proof_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_FL_PROOF_DIR"] = str(args.proof_dir / "smoke-capi")
    local_rank, rank, world = [
        int(os.environ[name]) for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    ]
    groups = validate_groups(json.loads(args.groups_file.read_text()), world)

    import torch
    import torch_npu  # noqa: F401 -- registers NPU backend
    import torch.distributed as dist

    torch.npu.set_device(local_rank)
    from sglang.srt.plugins import load_plugins
    import sglang_fl

    load_plugins()
    if not sglang_fl.is_plugin_active():
        raise RuntimeError("SG did not activate the plugin through its entry point")
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    result_path = args.proof_dir / f"smoke-{socket.gethostname()}-rank{rank}.jsonl"

    def emit(event, **fields):
        record = dict(
            event=event,
            pid=os.getpid(),
            rank=rank,
            world_size=world,
            host=socket.gethostname(),
            **fields,
        )
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with result_path.open("a") as out:
            out.write(line + "\n")

    dist.init_process_group("gloo", init_method="env://", timeout=timedelta(minutes=10))
    from sglang.srt.distributed import parallel_state

    srt = Path(parallel_state.__file__).resolve().parent.parent
    sources = {}
    for relative in (
        "distributed/parallel_state.py",
        "distributed/device_communicators/npu_communicator.py",
        "hardware_backend/npu/graph_runner/npu_graph_runner.py",
    ):
        path = srt / relative
        sources[str(path)] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    emit("runtime_sources", files=sources, graph_runner_executed=False)
    manifests = [None] * world
    dist.all_gather_object(manifests, groups)
    if any(manifest != groups for manifest in manifests):
        raise RuntimeError("Ranks received different group partitions")
    for name in ("tp", "attention_tp"):
        group = GroupCoordinator(
            group_ranks=groups[name],
            local_rank=local_rank,
            torch_distributed_backend="hccl",
            use_pynccl=False,
            use_pymscclpp=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_all_reduce=False,
            use_hpu_communicator=False,
            use_xpu_communicator=False,
            use_npu_communicator=True,
            group_name=name,
        )
        # Do not mask a failure with a potentially blocking collective destructor.
        # torchrun terminates peers on nonzero exit. Ordered teardown is success-only.
        run_group(torch, dist, group, args.mode, emit)
        dist.barrier()
        group.destroy()
    emit("smoke_passed", mode=args.mode, groups=groups)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
