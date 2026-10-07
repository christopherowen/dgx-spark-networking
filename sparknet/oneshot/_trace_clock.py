"""GPU-minus-CPU clock offset on this node, for ``sparknet.oneshot.trace``.

The kernel stamps trace records with the GPU's ``%globaltimer`` and the proxy thread with
``CLOCK_MONOTONIC_RAW``. A one-thread probe kernel answers host pings through pinned
memory: the host writes round ``i`` and notes the CPU time before the write and after it
sees the GPU's reply, the GPU replies with its timer as soon as it sees the round. The
round with the shortest round trip bounds the offset most tightly: offset = GPU reply -
midpoint of the CPU times, give or take half that round trip.
"""

from __future__ import annotations

import functools
import time

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import numpy as np
import torch
from cutlass import Int32, Int64, Uint32

from ._compile import compile_kernel
from ._cute_intrinsics import globaltimer, ld_relaxed_sys_u32, st_relaxed_sys_u64

ROUNDS = 400
SPIN_LIMIT = 50_000_000


class _ClockProbe:
    @cute.jit
    def __call__(self, base: Int64, rounds: Int32, spin_limit: Int32, stream: cuda.CUstream) -> None:
        self.kernel(base, rounds, spin_limit).launch(grid=(1, 1, 1), block=[32, 1, 1], cluster=(1, 1, 1),
                                                     stream=stream)

    @cute.kernel
    def kernel(self, base: Int64, rounds: Int32, spin_limit: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) == Int32(0):
            round_ = Int32(1)
            while round_ <= rounds:
                polls = Int32(0)
                waiting = Int32(1)
                while waiting == Int32(1):
                    if ld_relaxed_sys_u32(base) == Uint32(round_):
                        waiting = Int32(0)
                    else:
                        polls = polls + Int32(1)
                        if polls > spin_limit:
                            waiting = Int32(0)
                st_relaxed_sys_u64(base + Int64(8) * Int64(round_), globaltimer())
                round_ = round_ + Int32(1)


@functools.cache
def _launcher():
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return compile_kernel(_ClockProbe(), 16, 1, 1, stream, name="oneshot.trace_clock", cache_key=("trace_clock",))


def calibrate(device: torch.device, rounds: int = ROUNDS) -> tuple[int, int]:
    """(GPU minus CPU clock in ns, half the best round trip in ns) for ``device``'s node."""
    with torch.cuda.device(device):
        buffer = torch.zeros(8 * (rounds + 1), dtype=torch.uint8, pin_memory=True)
        words32 = buffer.numpy().view(np.uint32)
        words64 = buffer.numpy().view(np.uint64)
        launcher = _launcher()
        stream = torch.cuda.Stream(device)
        launcher(buffer.data_ptr(), rounds, SPIN_LIMIT, cuda.CUstream(stream.cuda_stream))
        samples = []
        for round_ in range(1, rounds + 1):
            before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            words32[0] = round_
            deadline = before + 1_000_000_000
            while words64[round_] == 0:
                if time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) > deadline:
                    raise RuntimeError("trace clock probe did not answer")
            after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            samples.append((int(words64[round_]), before, after))
        stream.synchronize()
    gpu, before, after = min(samples, key=lambda s: s[2] - s[1])
    return gpu - (before + after) // 2, (after - before) // 2


__all__ = ["calibrate"]
