"""One rank of a fail-stop scenario, in a process of its own.

A collective wait that times out stops the GPU (``trap``), which ends this
process's CUDA context, so the scenario cannot run inside the pytest process
that owns the suite's runtime. ``test_oneshot_gpu.py`` starts this script on
every rank (torchrun's ``RANK``, ``WORLD_SIZE`` and ``MASTER_ADDR``, plus
``FAILSTOP_PORT``) and reads the one JSON line it prints:

- ``error``: what CUDA raised once the GPU stopped (None if nothing failed);
- ``failure``: the runtime's failure report, read from pinned host memory after
  the stop;
- ``seconds``: from the injected fault to the error.

usage: failstop_child.py eager|graph
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

FAULT_TIMEOUT_MS = 2000
CAPTURE_TIMEOUT_MS = 60000


def _runtime(timeout_ms: int):
    from sparknet import oneshot as roce

    os.environ["SPARKNET_ROCE_TIMEOUT_MS"] = str(timeout_ms)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rt = roce.AllReduce.from_exchange_group(exchange_group=dist.group.WORLD, device=device, max_size=1 << 20)
    rt.prepare((torch.bfloat16,))
    return rt


def _until_stopped(step) -> tuple[str | None, float]:
    """Run ``step`` until CUDA reports the stopped GPU (at most three times)."""
    started = time.monotonic()
    try:
        for _ in range(3):
            step()
            torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001 - the stopped GPU's error is the result
        return str(error)[:400], time.monotonic() - started
    return None, time.monotonic() - started


def eager(rank: int, world: int) -> dict:
    """Rank 1 stops its proxy: every rank's wait for it (or for a rank that
    stopped) times out and stops that rank's GPU."""
    rt = _runtime(FAULT_TIMEOUT_MS)
    x = torch.ones(4096, dtype=torch.bfloat16, device=rt.device)
    out = rt.all_reduce(x)
    torch.cuda.synchronize()
    healthy = rt.failure() is None and torch.equal(out, x * world)
    dist.barrier()
    if rank == 1:
        rt._proxy.stop()
    dist.barrier()
    error, seconds = _until_stopped(lambda: rt.all_reduce(x))
    return {"healthy_before": healthy, "error": error, "seconds": seconds, "failure": rt.failure()}


def graph(rank: int, world: int) -> dict:
    """Captured with a long timeout, lowered after capture: replays stop within
    the lowered one, so kernels read the timeout at run time."""
    rt = _runtime(CAPTURE_TIMEOUT_MS)
    static = torch.ones(4096, dtype=torch.bfloat16, device=rt.device)
    stream = torch.cuda.Stream(device=rt.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rt.all_reduce(static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured, stream=stream), rt.capture(stream=stream):
        o1 = rt.all_reduce(static)
        o2 = rt.all_reduce(o1)
        o3 = rt.all_reduce(o2)
    torch.cuda.synchronize()
    dist.barrier()
    captured.replay()
    torch.cuda.synchronize()
    healthy = rt.failure() is None and torch.equal(o3, static * world**3)
    rt.set_timeout_ms(FAULT_TIMEOUT_MS)
    dist.barrier()
    if rank == 1:
        rt._proxy.stop()
    dist.barrier()
    error, seconds = _until_stopped(captured.replay)
    return {"healthy_before": healthy, "error": error, "seconds": seconds, "failure": rt.failure()}


def main(argv: list[str]) -> int:
    scenario = argv[0]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['FAILSTOP_PORT']}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    report = {"rank": rank, "scenario": scenario, "timeout_ms": FAULT_TIMEOUT_MS}
    report.update({"eager": eager, "graph": graph}[scenario](rank, world))
    print(json.dumps(report), flush=True)
    # The CUDA context is gone: leave without teardown that would touch it.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
