"""GPU-minus-CPU clock offset on this node, for ``sparknet.oneshot.trace``.

The kernel stamps trace records with the GPU's ``%globaltimer`` and the proxy thread with
``CLOCK_MONOTONIC_RAW``. A one-thread probe kernel answers host pings through pinned
memory: the host writes round ``i`` and notes the CPU time before the write and after it
sees the GPU's reply, the GPU replies with its timer as soon as it sees the round. The
round with the shortest round trip bounds the offset most tightly: offset = GPU reply -
midpoint of the CPU times, give or take half that round trip.
"""

# No `from __future__ import annotations`: TileLang evaluates the prim_func's
# annotations, and its symbolic shape (a local of the jitted function) must resolve.
import functools
import time

import numpy as np
import torch

ROUNDS = 400
SPIN_LIMIT = 50_000_000

# One thread answers the host's rounds: wait (bounded) for round i in the first
# word, then store %globaltimer in word i, all at system scope on pinned memory.
_SOURCE = r"""
__device__ __forceinline__ void roce_clock_probe(long long base, int rounds, int spin_limit) {
    for (int round = 1; round <= rounds; round++) {
        for (int polls = 0; polls <= spin_limit; polls++) {
            unsigned int seen;
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(base) : "memory");
            if (seen == (unsigned int)round) {
                break;
            }
        }
        unsigned long long now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(base + 8LL * round), "l"(now) : "memory");
    }
}
"""


@functools.cache
def _launcher():
    """The probe kernel: thread 0 of one block runs ``roce_clock_probe``. A small CUDA
    tensor binds the launch to the current device and stream."""
    import tilelang
    import tilelang.language as T

    from ._oneshot_tilelang import _pass_configs

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n = T.dynamic("n")

        @T.prim_func
        def clock_probe(anchor: T.Tensor((n,), T.int32), base: T.int64, rounds: T.int32, spin_limit: T.int32):
            with T.Kernel(1, threads=32):
                T.import_source(_SOURCE)
                tx = T.get_thread_binding()
                if tx == 0:
                    T.call_extern("roce_clock_probe", base, rounds, spin_limit, dtype="handle")

        return clock_probe

    return program()


def calibrate(device: torch.device, rounds: int = ROUNDS) -> tuple[int, int]:
    """(GPU minus CPU clock in ns, half the best round trip in ns) for ``device``'s node."""
    with torch.cuda.device(device):
        buffer = torch.zeros(8 * (rounds + 1), dtype=torch.uint8, pin_memory=True)
        words32 = buffer.numpy().view(np.uint32)
        words64 = buffer.numpy().view(np.uint64)
        launcher = _launcher()
        anchor = torch.zeros(1, dtype=torch.int32, device=device)
        stream = torch.cuda.Stream(device)
        with torch.cuda.stream(stream):
            launcher(anchor, buffer.data_ptr(), rounds, SPIN_LIMIT)
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
