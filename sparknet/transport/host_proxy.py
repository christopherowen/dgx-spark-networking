"""The host-proxy transport: a C thread over libibverbs posts every RDMA write.

This is the hardware-qualified path. ``sparknet.rocenante._proxy.Proxy`` is the
implementation; this module names it at the transport boundary and reports
its capability (a C compiler, the verbs headers and an active RDMA device).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import TransportCapability


def host_proxy_capability() -> TransportCapability:
    """Can the host proxy be built and run here? Read-only checks."""
    reasons = []
    if not any(shutil.which(c) for c in ("gcc", "cc", "clang")):
        reasons.append("no C compiler (gcc, cc or clang) on PATH")
    if not Path("/usr/include/infiniband/verbs.h").exists():
        reasons.append("libibverbs headers missing (/usr/include/infiniband/verbs.h)")
    devices = sorted(p.name for p in Path("/sys/class/infiniband").glob("*")) if Path("/sys/class/infiniband").exists() else []
    if not devices:
        reasons.append("no RDMA device under /sys/class/infiniband")
    return TransportCapability(
        name="host-proxy",
        available=not reasons,
        reasons=tuple(reasons),
        details={"rdma_devices": devices},
    )


def HostProxyTransport(**kwargs):  # noqa: N802 - factory with the class's name
    """Construct the host proxy; importing it pulls in ctypes only, not torch."""
    from sparknet.rocenante._proxy import Proxy

    return Proxy(**kwargs)


__all__ = ["HostProxyTransport", "host_proxy_capability"]
