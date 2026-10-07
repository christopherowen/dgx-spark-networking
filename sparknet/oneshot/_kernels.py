"""Kernel family selection: which generated kernels carry the one-shot protocol.

Two families implement the same protocol over the same pinned region, proxy
and launcher signature: the vendored CuTe DSL kernels (``cute``) and the
TileLang kernels (``tilelang``), which inject ``_device.py``. They are meant
to be bit-identical; the GPU test checks it. ``SPARKNET_ROCE_KERNELS``
selects the family for a process, the runtime's ``kernels`` keyword for one
runtime, and every rank of a group must agree (the setup handshake checks).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable

ENV_KERNELS = "SPARKNET_ROCE_KERNELS"
FAMILIES = ("cute", "tilelang")
DEFAULT_FAMILY = "cute"
PACK_BYTES = 16


@dataclass(frozen=True)
class Launch:
    """One collective launch, in the terms both kernel families take.

    ``input`` and ``output`` are flat ``int32`` views of the 16-byte-aligned
    input and output bytes; ``region`` is the pinned host region (``uint8``)
    and the offsets locate its receive slots, flags, send slots and control
    record; ``counters`` is the device-resident epoch, staging and tail
    counters and the poison word, addressed by index. A TileLang kernel takes
    the tensors as buffers (which also binds its device and stream); the CuTe
    launcher reads their addresses.
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
    poison_index: int
    spin_limit: int
    grid_x: int
    row_packs: int = 0  # all-gather only: packs per row of the shard (dim-0 gathers: the shard's packs)
    trace_base: int = 0  # all-reduce trace file address (pinned, device-visible), 0 when untraced


def family(explicit: str | None = None) -> str:
    """The selected family: the explicit choice, else ``SPARKNET_ROCE_KERNELS``, else ``cute``."""
    chosen = explicit or os.environ.get(ENV_KERNELS) or DEFAULT_FAMILY
    if chosen not in FAMILIES:
        raise ValueError(f"unknown one-shot kernel family {chosen!r}; choose one of {FAMILIES}")
    return chosen


def reduce_launcher(kernels: str, *key) -> Callable[..., None]:
    """The compiled all-reduce launcher of ``kernels`` for ``key`` (dtype, world, rank, geometry)."""
    return importlib.import_module(f"sparknet.oneshot._oneshot_{kernels}").get_launcher(*key)


def gather_launcher(kernels: str, *key) -> Callable[..., None]:
    """The compiled all-gather launcher of ``kernels`` for ``key`` (world, rank, geometry)."""
    return importlib.import_module(f"sparknet.oneshot._allgather_{kernels}").get_launcher(*key)


__all__ = ["DEFAULT_FAMILY", "ENV_KERNELS", "FAMILIES", "Launch", "PACK_BYTES", "family", "gather_launcher", "reduce_launcher"]
