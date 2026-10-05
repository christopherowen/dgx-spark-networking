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
from typing import Callable

ENV_KERNELS = "SPARKNET_ROCE_KERNELS"
FAMILIES = ("cute", "tilelang")
DEFAULT_FAMILY = "cute"
PACK_BYTES = 16


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


__all__ = ["DEFAULT_FAMILY", "ENV_KERNELS", "FAMILIES", "PACK_BYTES", "family", "gather_launcher", "reduce_launcher"]
