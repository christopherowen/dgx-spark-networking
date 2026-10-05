"""GPU-initiated transport over DOCA GPUNetIO: staged, not implemented.

Design (see ``docs/gpudirect-roadmap.md``):

1. **CPU-proxy GPUNetIO** (``DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY``):
   the collective kernel builds the RDMA write and flag WQEs itself, right
   after staging, and publishes them with a system-scope release; a host
   thread only rings the NIC doorbell. This removes the proxy's doorbell
   poll, stripe arithmetic and posting from the critical path. Qualified on
   every neighbour link of the four Sparks in the spark-ds41f
   ``2026-10-03-relay-progress`` experiment (ping-pong 4.1 to 7 us), with the
   patches under ``native/gpunetio``.
2. **GPU doorbell** (``NIC_HANDLER_GPU_SM_DB``): the kernel rings the NIC's
   UAR itself. The first attempt on the Sparks ended in a host-level failure
   (dgx1 rebooted, dgx2's GPU needed a reboot); it needs a separate
   host-level investigation before any further attempt.
3. **Device-memory registration** (``ibv_reg_dmabuf_mr`` or ``nvidia-peermem``):
   lets slots live in device memory instead of pinned host memory. rdma-core
   50 on the fleet exports the symbol; whether the GB10 driver exports a
   dma-buf for a device allocation is what ``sparknet probe gpudirect`` checks.

The transport keeps ``Geometry`` and the wire layout of the host proxy so the
kernels, flags and the relay semantics are unchanged; only who writes the
WQEs and who rings the doorbell moves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import TransportCapability

ENV_GPUNETIO_DIR = "SPARKNET_GPUNETIO_DIR"
PINNED_GPUNETIO_REVISION = "586453728bcab2d4c50574924dc6cf43543c9ed4"


def gpunetio_capability(root: str | os.PathLike[str] | None = None) -> TransportCapability:
    """Is a built GPUNetIO library present? Read-only; does not load it."""
    base = Path(root or os.environ.get(ENV_GPUNETIO_DIR, "")) if (root or os.environ.get(ENV_GPUNETIO_DIR)) else None
    reasons = []
    details: dict[str, Any] = {"revision": PINNED_GPUNETIO_REVISION}
    if base is None:
        reasons.append(f"{ENV_GPUNETIO_DIR} is not set (path of the built native/gpunetio tree)")
    else:
        details["root"] = str(base)
        lib = next(iter(sorted(base.glob("lib/libdoca_gpunetio*.so*"))), None)
        if lib is None:
            reasons.append("libdoca_gpunetio shared library not built under lib/")
        else:
            details["library"] = str(lib)
        header = base / "include" / "doca_gpunetio_device.h"
        if not header.exists():
            reasons.append("GPUNetIO device headers missing")
    return TransportCapability(
        name="gpunetio-cpu-proxy", available=not reasons, reasons=tuple(reasons), details=details
    )


class GpuNetIOTransport:
    """Placeholder that fails loudly: selecting it is a decision, never a fallback."""

    topology = "direct"
    traffic_class = 0

    def __init__(self, **geometry: Any) -> None:
        capability = gpunetio_capability()
        raise NotImplementedError(
            "GpuNetIOTransport is staged, not implemented; "
            + ("; ".join(capability.reasons) if capability.reasons else "see docs/gpudirect-roadmap.md")
        )


__all__ = ["ENV_GPUNETIO_DIR", "GpuNetIOTransport", "PINNED_GPUNETIO_REVISION", "gpunetio_capability"]
