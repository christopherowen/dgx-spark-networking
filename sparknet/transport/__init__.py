"""The RDMA transport boundary of the RoCEnante protocol.

The GPU side of the protocol (pinned slots, per-lane sequence flags, the
doorbell, the device-resident epoch, fixed-rank reduction) does not depend on
who posts the RDMA writes. Today the host proxy does (``HostProxyTransport``,
a C thread over libibverbs). The GPU-initiated path (``GpuNetIOTransport``)
keeps the same geometry and wire layout and replaces the posting side; see
``docs/gpudirect-roadmap.md`` for its stages and what each one still needs.
"""

from .base import Geometry, Transport, TransportCapability
from .host_proxy import HostProxyTransport
from .gpunetio import GpuNetIOTransport, gpunetio_capability

__all__ = [
    "Geometry",
    "GpuNetIOTransport",
    "HostProxyTransport",
    "Transport",
    "TransportCapability",
    "gpunetio_capability",
]
