#!/usr/bin/env python3
"""The smallest engine that uses the library the way a serving recipe should.

One process per Spark, launched with the environment that
``sparknet topology render <nodes.json> <node> --transport ... --profile ...``
prints for that node::

    python3 examples/minimal_engine.py --rank 0 --world-size 4 --master-addr 192.0.2.1 --master-port 29650

It initializes torch.distributed with NCCL for CUDA tensors and gloo for the
setup exchange, builds the one-shot runtime from the policy in the
environment, prepares the launchers before any CUDA graph, runs a few decode
steps eagerly and from a captured graph, dispatching every collective
through the policy, and checks health after each step's host sync.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from sparknet import oneshot
from sparknet.policy import CollectivePolicy

HIDDEN = 5120  # DeepSeek V4.1 Flash hidden width; a 6-token decode step is 60 KiB in BF16


def dispatch_all_reduce(runtime, policy, x: torch.Tensor) -> torch.Tensor:
    """The policy decides, every rank alike, from dtype, shape and byte size."""
    nbytes = x.numel() * x.element_size()
    dtype = str(x.dtype).replace("torch.", "")
    if policy.all_reduce_backend(nbytes, dtype, contiguous=x.is_contiguous()) == "oneshot":
        return runtime.all_reduce(x)
    out = x.clone()
    dist.all_reduce(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=6)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=args.rank, world_size=args.world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}", timeout=timedelta(seconds=120),
    )
    cpu_group = dist.new_group(backend="gloo")  # the setup exchange; never a torch NCCL communicator

    policy = CollectivePolicy.from_environment(dict(os.environ))
    runtime = oneshot.AllReduce.from_exchange_group(
        exchange_group=cpu_group, device=device,
        max_size=policy.all_reduce_capacity_bytes, max_gather_bytes=policy.all_gather_shard_bytes,
    )
    runtime.prepare((torch.bfloat16,), padded_gather=True)  # before any capture; no JIT in a step
    if args.rank == 0:
        print("one-shot runtime:", {k: runtime.stats()[k] for k in ("topology", "hcas", "stripe_count", "dispatch_max_bytes")})

    # A decode step: one all-reduce per layer on [tokens, HIDDEN] BF16 activations.
    hidden = torch.full((args.tokens, HIDDEN), args.rank + 1, dtype=torch.bfloat16, device=device)
    expected = float(args.world_size * (args.world_size + 1) // 2)
    for _ in range(3):
        out = dispatch_all_reduce(runtime, policy, hidden)
        torch.cuda.synchronize()        # the step's own device-to-host sync
        runtime.check_health()          # fail-stop: raises if any rank's wait timed out
        assert torch.all(out == expected), "eager all-reduce mismatch"

    # The same step captured once and replayed; every collective on the capture stream.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
        captured = dispatch_all_reduce(runtime, policy, hidden)
    for step in range(1, 4):
        hidden.fill_(args.rank + 1 + step)
        graph.replay()
        torch.cuda.synchronize()
        runtime.check_health()
        assert torch.all(captured == expected + step * args.world_size), "graph replay mismatch"

    oneshot.freeze_kernel_resolution("warm-up complete")  # any later compile is a bug
    if args.rank == 0:
        print("ok:", runtime.stats()["ops_posted"], "one-shot operations posted")
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
