"""Fail-stop qualification: one rank of a fault scenario, in a process of its own.

Rank 1 stops its proxy; every rank's collective wait for it (or for a rank that
stopped) must time out and stop the GPU (``trap``) within about one timeout,
and the runtime must name the rank it waited for. Stopping the GPU ends this
process's CUDA context, so the scenario runs in a process of its own, on every
rank at once (the GPU suite starts it per rank; a fleet runs it as a fabric
script). Prints one JSON line and exits 0 when the rank behaved:

- ``error``: what CUDA (or the runtime's own check) raised once the GPU stopped;
- ``failure``: the runtime's failure report, read from pinned host memory;
- ``seconds``: from the injected fault to the error;
- ``passed`` and ``problems``: the verdict.

Scenarios: ``eager``; ``graph`` (captured under a 60 s timeout, lowered to 2 s
after capture: kernels read the timeout when they wait).

usage: python -m sparknet.probe.failstop eager|graph --rank R --world-size W
       --master-addr A --master-port P [other probe arguments are ignored]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

FAULT_TIMEOUT_MS = 2000
CAPTURE_TIMEOUT_MS = 60000
# A stopped rank's report: its own timed-out wait, and on the ring possibly a
# relay neighbour's proxy that saw the stopped peer first.
FAIL_STOP = r"saw nothing from rank \d+|proxy failed"


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


def problems(report: dict) -> list[str]:
    """Why a rank's report is not a qualified fail-stop (empty when it is)."""
    timeout_s = report["timeout_ms"] / 1000
    found = []
    if not report["healthy_before"]:
        found.append("the runtime was not healthy before the fault")
    if report["error"] is None:
        found.append("nothing stopped: the collectives kept running past the fault")
    if not re.search(FAIL_STOP, report["failure"] or ""):
        found.append("the failure report does not name the stalled peer or relay")
    # One timeout for a rank waiting on the stopped one, two for a rank that first
    # completed on the others' payloads.
    if not 0.9 * timeout_s <= report["seconds"] <= 3 * timeout_s + 2:
        found.append(f"stopped after {report['seconds']:.2f} s, not within the {timeout_s:g} s timeout")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=("eager", "graph"))
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    args, _ = parser.parse_known_args(argv)
    if args.world_size < 2:
        parser.error("a fault scenario needs at least two ranks")
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(seconds=120),
    )
    report = {"rank": args.rank, "scenario": args.scenario, "timeout_ms": FAULT_TIMEOUT_MS}
    report.update({"eager": eager, "graph": graph}[args.scenario](args.rank, args.world_size))
    report["problems"] = problems(report)
    report["passed"] = not report["problems"]
    print(json.dumps(report), flush=True)
    # The CUDA context is gone: leave without teardown that would touch it.
    os._exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
