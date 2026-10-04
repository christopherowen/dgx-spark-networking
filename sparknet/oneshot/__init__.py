"""``sparknet.oneshot``: One-shot RDMA collectives for multi-node tensor parallelism on DGX Spark.

Target: Spark nodes joined by their ConnectX-7 200 GbE ports without a switch,
one GPU per node. The GB10's unified memory lets the NIC RDMA-write straight
into pinned host memory that the GPU kernel then reads in place, so the
protocol needs no GPUDirect RDMA (dmabuf or peermem) to run; the transport
boundary in ``sparknet.transport`` is where a GPU-initiated path replaces the
host proxy later.

``AllReduce`` is the runtime (``from_exchange_group``, ``prepare``,
``should_allreduce``, ``all_reduce``, ``should_all_gather``, ``all_gather``,
``capture``, ``check_health``, ``poisoned``, ``stats``, ``close``).
See ``runtime.py`` for the protocol and the contract, and ``docs/oneshot.md``.

Importing this package imports torch and the CuTe DSL.
"""

from __future__ import annotations

from ._compile import (
    KernelResolutionFrozenError,
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    thaw_kernel_resolution,
)
from .runtime import (
    API_VERSION,
    DEFAULT_MAX_GATHER_BYTES,
    DEFAULT_MAX_SIZE,
    SUPPORTED_DTYPES,
    SUPPORTED_WORLD_SIZES,
    TOPOLOGIES,
    RoceOneshotAllReduce as AllReduce,
    default_gid_index,
    discover_hcas,
    is_supported,
)

__all__ = [
    "API_VERSION",
    "AllReduce",
    "DEFAULT_MAX_GATHER_BYTES",
    "DEFAULT_MAX_SIZE",
    "KernelResolutionFrozenError",
    "SUPPORTED_DTYPES",
    "SUPPORTED_WORLD_SIZES",
    "TOPOLOGIES",
    "default_gid_index",
    "discover_hcas",
    "freeze_kernel_resolution",
    "is_supported",
    "kernel_resolution_frozen",
    "thaw_kernel_resolution",
]
