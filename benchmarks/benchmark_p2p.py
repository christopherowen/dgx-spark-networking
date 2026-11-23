"""Latency and bandwidth of one-shot RoCE P2P versus torch.distributed (NCCL).

Launch with torchrun on two directly cabled or switched Spark nodes::

    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29652 \\
        benchmarks/benchmark_p2p.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def _time_pingpong(fn, warmup: int, iters: int) -> tuple[float, float]:
    """Return (median_us_per_oneway, gbps) over ``iters`` round trips."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6 / 2.0)  # one-way latency
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sparknet.oneshot.P2P vs NCCL isend/irecv")
    parser.add_argument(
        "--sizes-kib",
        type=str,
        default="4,64,300,1200,4096,16384,65536",
        help="Comma-separated payload sizes in KiB",
    )
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--slot-mib", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"benchmark_p2p.py expects world_size == 2, got {world}")
    peer = 1 - rank
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    from sparknet import oneshot

    rt = oneshot.P2P.from_exchange_group(
        exchange_group=cpu_group,
        device=device,
        slot_bytes=args.slot_mib << 20,
        pack_bytes=16 << 20,
    )
    rt.prepare()

    sizes = [int(s) * 1024 for s in args.sizes_kib.split(",") if s.strip()]
    if rank == 0:
        print(
            f"P2P benchmark: hcas={rt.hca_names} stripes={rt.stripe_count} "
            f"slot={rt.slot_bytes >> 20} MiB iters={args.iters}"
        )
        print(
            f"{'Size':>10} | {'NCCL eager':>11} | {'P2P eager':>11} | {'P2P graph':>11} | "
            f"{'NCCL GB/s':>9} | {'P2P GB/s':>9} | {'Speedup':>8}"
        )
        print("-" * 86)

    for nbytes in sizes:
        numel = nbytes // 2
        torch.manual_seed(1234 + rank)
        src_t = torch.randn(numel, dtype=torch.bfloat16, device=device)
        dst_t = torch.empty_like(src_t)
        ref_t = torch.empty_like(src_t)

        # Bit-exactness gate against NCCL
        def nccl_roundtrip(a=src_t, b=ref_t):
            if rank == 0:
                w1 = dist.isend(a, dst=peer)
                w2 = dist.irecv(b, src=peer)
            else:
                w2 = dist.irecv(b, src=peer)
                w1 = dist.isend(a, dst=peer)
            w1.wait()
            w2.wait()

        def p2p_roundtrip(a=src_t, b=dst_t):
            if rank == 0:
                rt.send(a, dst=peer)
                rt.recv(b, src=peer)
            else:
                rt.recv(b, src=peer)
                rt.send(a, dst=peer)

        nccl_roundtrip()
        p2p_roundtrip()
        torch.cuda.synchronize()
        rt.check_health()
        assert torch.equal(dst_t, ref_t), f"P2P mismatch at {nbytes} B"

        nccl_med, _ = _time_pingpong(nccl_roundtrip, args.warmup, args.iters)
        p2p_med, _ = _time_pingpong(p2p_roundtrip, args.warmup, args.iters)

        # CUDA-graph replay timing for payloads <= 16 MiB
        if nbytes <= (16 << 20):
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                p2p_roundtrip()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream), rt.capture(stream=stream):
                p2p_roundtrip()
            p2p_graph_med, _ = _time_pingpong(g.replay, args.warmup, args.iters)
            graph_str = f"{p2p_graph_med:8.1f} us"
        else:
            p2p_graph_med = p2p_med
            graph_str = "       n/a"

        # Post-timing exactness check
        dst_t.zero_()
        p2p_roundtrip()
        torch.cuda.synchronize()
        rt.check_health()
        assert torch.equal(dst_t, ref_t), f"post-timing P2P mismatch at {nbytes} B"

        nccl_gbps = (nbytes / 1e9) / (nccl_med / 1e6)
        p2p_gbps = (nbytes / 1e9) / (p2p_med / 1e6)
        speedup = nccl_med / p2p_graph_med
        label = f"{nbytes // 1024} KiB" if nbytes < (1 << 20) else f"{nbytes // (1 << 20)} MiB"
        if rank == 0:
            print(
                f"{label:>10} | {nccl_med:8.1f} us | {p2p_med:8.1f} us | {graph_str} | "
                f"{nccl_gbps:9.2f} | {p2p_gbps:9.2f} | {speedup:7.2f}x"
            )

    rt.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        raise SystemExit("Run with torchrun --nnodes=2 --nproc-per-node=1 ...")
    main()
