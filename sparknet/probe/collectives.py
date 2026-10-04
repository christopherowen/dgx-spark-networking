#!/usr/bin/env python3
"""Model-free qualification of the collective policy on the actual fabric.

Run one rank per node (``sparknet probe render-command`` prints the bounded
container command). The probe initializes torch.distributed, constructs the
one-shot runtime from the environment the recipe will use, then checks the
exact policy: eligible all-reduces and all-gathers through one-shot, the
rest and every reduce-scatter through NCCL. It tests BF16/FP32, tiny,
unaligned and large messages, both sides of every dispatch boundary, eager
execution and CUDA graph replay with changing inputs, then optionally screens
steady graph latency with RDMA and port counters around every timed case.

A passing probe is a prerequisite for serving, not serving acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from datetime import timedelta


def _policy_from_environment():
    from sparknet.policy import CollectivePolicy

    return CollectivePolicy.from_environment(dict(os.environ))


class PolicyCommunicator:
    """The explicit two-backend policy over torch.distributed and the one-shot runtime."""

    def __init__(self, runtime, policy, device, world_size):
        import torch
        import torch.distributed as dist

        self._torch, self._dist = torch, dist
        self.runtime, self.policy, self.device, self.world_size = runtime, policy, device, world_size

    def _dtype_name(self, tensor) -> str:
        return str(tensor.dtype).replace("torch.", "")

    def all_reduce(self, tensor):
        nbytes = tensor.numel() * tensor.element_size()
        if (self.runtime is not None
                and self.policy.all_reduce_backend(nbytes, self._dtype_name(tensor), contiguous=tensor.is_contiguous()) == "oneshot"):
            if not self.runtime.should_allreduce(tensor):
                raise RuntimeError("policy and runtime eligibility disagree for all-reduce")
            return self.runtime.all_reduce(tensor)
        out = tensor.clone()
        self._dist.all_reduce(out)
        return out

    def all_gather(self, tensor, dim=0):
        nbytes = tensor.numel() * tensor.element_size()
        if (self.runtime is not None
                and self.policy.all_gather_backend(nbytes, self._dtype_name(tensor), dim=dim, ndim=tensor.dim(),
                                                   contiguous=tensor.is_contiguous()) == "oneshot"):
            if not self.runtime.should_all_gather(tensor, dim):
                raise RuntimeError("policy and runtime eligibility disagree for all-gather")
            return self.runtime.all_gather(tensor, dim=dim)
        parts = [self._torch.empty_like(tensor) for _ in range(self.world_size)]
        self._dist.all_gather(parts, tensor.contiguous())
        return self._torch.cat(parts, dim=dim)

    def reduce_scatter(self, tensor, dim=0):
        if dim != 0:
            raise ValueError("reduce-scatter is tested along dim 0")
        out = self._torch.empty((tensor.shape[0] // self.world_size,) + tuple(tensor.shape[1:]),
                                dtype=tensor.dtype, device=tensor.device)
        self._dist.reduce_scatter_tensor(out, tensor.contiguous())
        return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    from sparknet.topology.nodes import TRANSPORTS, roce_topology, uses_oneshot

    parser.add_argument("--transport", choices=TRANSPORTS, default=None,
                        help="defaults to oneshot-<SPARKNET_ROCE_TOPOLOGY>, or nccl-direct when VLLM_ENABLE_ROCE_ALLREDUCE=0")
    parser.add_argument("--benchmark", action="store_true", help="after correctness, screen steady graph collective latency")
    parser.add_argument("--counter-samples", action="store_true", help="record RDMA error deltas around each benchmark case")
    parser.add_argument("--port-samples", action="store_true", help="sample physical NIC bytes and buffer drops around each timed case")
    parser.add_argument("--lengths", type=int, nargs="+", default=[5120, 30720, 245760, 1048576])
    parser.add_argument("--numerics", action="store_true", help="cancellation-sensitive reduction-order fingerprints")
    parser.add_argument("--output", default="", help="write the JSON result here as well as to stdout")
    args = parser.parse_args(argv)
    if any(n < 64 or n > 5242880 or n % 8 for n in args.lengths):
        parser.error("benchmark lengths must be aligned and between 64 and 5242880")
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must be in the configured process group")

    import torch
    import torch.distributed as dist

    from sparknet.probe.counters import deltas, port_counters, rdma_error_counters

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    roce_requested = os.environ.get("VLLM_ENABLE_ROCE_ALLREDUCE", "1") != "0"
    topology = os.environ.get("SPARKNET_ROCE_TOPOLOGY") or os.environ.get("B12X_ROCE_TOPOLOGY") or "direct"
    transport = args.transport or (f"oneshot-{topology}" if roce_requested else "nccl-direct")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", world_size=args.world_size, rank=args.rank,
        init_method=f"tcp://{args.master_addr}:{args.master_port}", timeout=timedelta(seconds=90),
    )
    cpu_group = dist.new_group(backend="gloo")
    runtime = None
    policy = None
    if uses_oneshot(transport):
        from sparknet import oneshot

        policy = _policy_from_environment()
        runtime = oneshot.AllReduce.from_exchange_group(
            exchange_group=cpu_group, device=device,
            max_size=policy.all_reduce_capacity_bytes, max_gather_bytes=policy.all_gather_shard_bytes,
        )
        if runtime.topology != roce_topology(transport):
            raise RuntimeError(f"runtime topology {runtime.topology} differs from the {transport} routing mode")
        if runtime.dispatch_max_bytes != policy.all_reduce_dispatch_bytes:
            raise RuntimeError("runtime did not apply the requested dispatch limit")
        runtime.prepare((torch.bfloat16, torch.float32, torch.float16), padded_gather=True)
    comm = PolicyCommunicator(runtime, policy, device, args.world_size)
    capture_context = (lambda stream: runtime.capture(stream=stream)) if runtime else (lambda stream: nullcontext())

    total = args.world_size * (args.world_size + 1) // 2
    control = torch.tensor([args.rank + 1], device=device, dtype=torch.float32)
    dist.all_reduce(control)
    torch.testing.assert_close(control, torch.full_like(control, total), rtol=0, atol=0)
    dist.broadcast(control, src=0)

    checks = []
    before = runtime.stats()["ops_posted"] if runtime else 0
    for dtype in (torch.bfloat16, torch.float32):
        lengths = {1, 17, 1024, 2 * 1024 * 1024}
        if runtime:
            for limit in (runtime.max_size, runtime.max_gather_bytes, runtime.dispatch_max_bytes):
                # Exercise the exact dispatch boundary and both adjacent 16-byte packs.
                lengths.update((limit // dtype.itemsize + offset) for offset in (-16 // dtype.itemsize, 0, 16 // dtype.itemsize))
        for length in sorted(lengths):
            pattern = (torch.arange(length, device=device) % 7).to(dtype)
            local = pattern + args.rank + 1
            scattered = torch.cat([local + 8 * peer for peer in range(args.world_size)])

            def collect():
                return (comm.all_reduce(local), comm.all_gather(local, dim=0), comm.reduce_scatter(scattered, dim=0))

            def verify(outputs, increment=0):
                ar, ag, rs = outputs
                torch.cuda.synchronize()
                torch.testing.assert_close(ar, pattern * args.world_size + total + increment * args.world_size, rtol=0, atol=0)
                torch.testing.assert_close(ag, torch.cat([pattern + peer + 1 + increment for peer in range(args.world_size)]), rtol=0, atol=0)
                expected = pattern * args.world_size + total + (8 * args.rank + increment) * args.world_size
                torch.testing.assert_close(rs, expected, rtol=0, atol=0)

            verify(collect())  # warm every size before capture so connection setup is outside it
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream), capture_context(stream):
                outputs = collect()
            for step in range(1, 5):
                local.add_(1)
                scattered.add_(1)
                graph.replay()
                verify(outputs, step)
            if runtime:
                runtime.check_health()
            checks.append({"dtype": str(dtype), "elements_per_rank": length, "eager": True, "graph_replays": 4})
            del graph, outputs, local, scattered, pattern

    numerical_checks = []
    if args.numerics:
        for dtype in (torch.bfloat16, torch.float32):
            base = torch.arange(64, dtype=torch.float32) % 8
            inputs = [(base + 1) * 16777216, (base + 1) / 16, -(base + 1) * 16777216, (base + 1) / 32]
            inputs = [v.to(dtype) for v in inputs][:args.world_size]
            reference = inputs[0].float()
            for v in inputs[1:]:
                reference = reference + v.float()
            reference = reference.to(dtype)
            for length in args.lengths:
                local = inputs[args.rank].to(device).repeat((length + 63) // 64)[:length].contiguous()
                output = comm.all_reduce(local)
                torch.cuda.synchronize()
                observed = output[:64].cpu()
                row = {"dtype": str(dtype), "elements_per_rank": length,
                       "prefix_sha256": hashlib.sha256(observed.view(torch.uint8).numpy().tobytes()).hexdigest(),
                       "rank_order_fp32_prefix_sha256": hashlib.sha256(reference.view(torch.uint8).numpy().tobytes()).hexdigest(),
                       "mismatched_reference_elements": int((observed != reference).sum()),
                       "max_abs_reference_difference": float((observed.float() - reference.float()).abs().max())}
                numerical_checks.append(row)
                print(json.dumps({"rank": args.rank, "numerical_check": row}), flush=True)

    timings = []
    if args.benchmark:
        for dtype in (torch.bfloat16, torch.float32):
            for length in args.lengths:
                local = torch.full((length,), args.rank + 1, device=device, dtype=dtype)
                scattered = torch.cat([local + 8 * p for p in range(args.world_size)])
                operations = {
                    "all_reduce": lambda: comm.all_reduce(local),
                    "all_gather": lambda: comm.all_gather(local, dim=0),
                    "reduce_scatter": lambda: comm.reduce_scatter(scattered, dim=0),
                }
                for name, operation in operations.items():
                    operation()
                    torch.cuda.synchronize()
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream), capture_context(stream):
                        outputs = [operation() for _ in range(16)]
                    for _ in range(4):
                        graph.replay()
                    torch.cuda.synchronize()
                    if args.counter_samples or args.port_samples:
                        dist.barrier(group=cpu_group)
                    counters_before = rdma_error_counters() if args.counter_samples else None
                    ports_before = port_counters() if args.port_samples else None
                    proxy_before = runtime.stats() if runtime and args.port_samples else None
                    samples = []
                    for _ in range(5):
                        dist.barrier(group=cpu_group)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(16):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / 256)
                    expected = (torch.full_like(local, total) if name == "all_reduce"
                                else torch.cat([torch.full_like(local, p + 1) for p in range(args.world_size)])
                                if name == "all_gather" else torch.full_like(local, total + 8 * args.rank * args.world_size))
                    for output in outputs:
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    row = {"dtype": str(dtype), "elements_per_rank": length, "operation": name,
                           "microseconds_per_call": samples, "calls_per_graph": 16, "replays_per_sample": 16}
                    if args.counter_samples or args.port_samples:
                        dist.barrier(group=cpu_group)
                    if counters_before is not None:
                        row["rdma_error_deltas"] = deltas(counters_before, rdma_error_counters())
                    if proxy_before is not None:
                        nbytes = local.numel() * local.element_size()
                        expected_custom = runtime is not None and (
                            (name == "all_reduce" and policy.all_reduce_backend(nbytes, str(dtype).replace("torch.", "")) == "oneshot")
                            or (name == "all_gather" and policy.all_gather_backend(nbytes, str(dtype).replace("torch.", ""), dim=0, ndim=1) == "oneshot"))
                        row["expected_backend"] = "oneshot" if expected_custom else "nccl"
                        proxy_after = runtime.stats()
                        row["proxy_payload_bytes"] = {h: after - before_ for h, after, before_ in zip(
                            proxy_after["hcas"], proxy_after["bytes_posted_per_hca"], proxy_before["bytes_posted_per_hca"])}
                        if bool(sum(row["proxy_payload_bytes"].values())) != expected_custom:
                            raise RuntimeError(f"{name}: actual transport counters disagree with the dispatch policy")
                    if ports_before is not None:
                        row["port_deltas"] = deltas(ports_before, port_counters())
                    if args.counter_samples or args.port_samples:
                        dist.barrier(group=cpu_group)
                    timings.append(row)
                    print(json.dumps({"rank": args.rank, "timing": row}), flush=True)
                    del graph, outputs
                del local, scattered

    proxy_stats = None
    if runtime:
        runtime.check_health()
        if runtime.stats()["ops_posted"] <= before:
            raise RuntimeError("no probe payload used the one-shot proxy")
        proxy_stats = runtime.stats()
    result = {"rank": args.rank, "world_size": args.world_size, "transport": transport,
              "policy": policy.__dict__ if policy else None, "proxy": proxy_stats, "passed": True,
              "checks": checks, "numerical_checks": numerical_checks, "timings": timings}
    text = json.dumps(result)
    print(text, flush=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    if runtime:
        runtime.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
