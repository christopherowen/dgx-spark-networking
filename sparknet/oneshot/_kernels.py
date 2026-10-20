"""The kernels that carry the one-shot protocol.

TileLang programs (``_oneshot_tilelang.py``, ``_allgather_tilelang.py``)
generated as CUDA with the protocol steps of ``_device.py``. Until 0.4.0 a
vendored CuTe DSL pair carried the same protocol bit for bit; one family is
qualified, maintained and run (``SPARKNET_ROCE_KERNELS`` is refused).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

PACK_BYTES = 16
REMOVED_KERNEL_SETTING = "SPARKNET_ROCE_KERNELS"


@dataclass(frozen=True)
class Launch:
    """One collective launch, in the terms the kernels take.

    ``input`` and ``output`` are flat ``int32`` views of the 16-byte-aligned
    input and output bytes; ``region`` is the pinned host region (``uint8``)
    and the offsets locate its receive slots, flags, send slots and control
    record; ``counters`` is the device-resident epoch, staging and tail
    counters and the rung and arrival words, addressed by index. A TileLang kernel takes
    the tensors as buffers (which also binds its device and stream).
    """

    input: Any
    output: Any
    size_packs: int
    nbytes: int
    region: Any
    recv_off: int
    flag_off: int
    send_off: int
    ctrl_off: int
    slot_bytes: int
    counters: Any
    stage_index: int
    tail_index: int
    rung_index: int  # set to the sequence by the block that rang the doorbell
    arrival_index: int  # set to the sequence by block 0 once every peer's flags arrived
    grid_x: int
    row_packs: int = 0  # all-gather only: packs per row of the shard (dim-0 gathers: the shard's packs)
    trace_base: int = 0  # all-reduce trace file address (pinned, device-visible), 0 when untraced


def kernel_setting_problems(environ: Mapping[str, str]) -> list[str]:
    """The family switch is gone: one message when it is still set."""
    if environ.get(REMOVED_KERNEL_SETTING):
        return [f"{REMOVED_KERNEL_SETTING} is no longer read: sparknet 0.4.0 runs only the TileLang "
                "kernels (the CuTe DSL pair they were bit-identical to is retired); unset it"]
    return []


def reduce_launcher(*key) -> Callable[..., None]:
    """The compiled all-reduce launcher for ``key`` (dtype, world, rank, geometry)."""
    from ._oneshot_tilelang import get_launcher

    return get_launcher(*key)


def gather_launcher(*key) -> Callable[..., None]:
    """The compiled all-gather launcher for ``key`` (world, rank, geometry)."""
    from ._allgather_tilelang import get_launcher

    return get_launcher(*key)


__all__ = ["Launch", "PACK_BYTES", "REMOVED_KERNEL_SETTING", "gather_launcher", "kernel_setting_problems",
           "reduce_launcher"]
