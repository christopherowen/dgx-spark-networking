"""Transport protocol and geometry shared by every RDMA posting implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Geometry:
    """Everything a transport needs to open its queues; identical on every rank except ``rank`` and routes."""

    world_size: int
    rank: int
    topology: str  # direct | ring4 | mesh4
    hca_names: tuple[str, ...]
    peer_hca_indices: tuple[tuple[int, ...], ...]
    stripe_count: int
    gid_index: int
    slot_bytes: int
    region_ptr: int
    region_bytes: int

    def validate(self) -> list[str]:
        problems = []
        if not 2 <= self.world_size <= 16:
            problems.append("world_size must be 2..16")
        if not 0 <= self.rank < self.world_size:
            problems.append("rank must belong to the group")
        if self.topology not in ("direct", "ring4", "mesh4"):
            problems.append(f"unknown topology {self.topology!r}")
        if self.topology != "direct" and self.world_size != 4:
            problems.append(f"{self.topology} requires four ranks")
        if not 1 <= len(self.hca_names) <= 4:
            problems.append("1..4 local HCAs")
        if len(self.peer_hca_indices) != self.world_size:
            problems.append("peer routes must cover every rank")
        if self.slot_bytes <= 0 or self.slot_bytes % 4096:
            problems.append("slot_bytes must be a positive multiple of 4096")
        return problems


@dataclass(frozen=True)
class TransportCapability:
    """What a transport can do on this host, decided without touching the fabric."""

    name: str
    available: bool
    reasons: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Transport(Protocol):
    """One rank's RDMA posting side of the one-shot protocol.

    Construction opens the HCAs and registers the pinned region; ``local_blob``
    and ``connect`` exchange queue-pair records through the caller's CPU
    process group; ``start`` begins posting on doorbells; ``failed``/``error``
    report fail-stop state; ``stats`` returns counters; ``close`` releases
    everything. The runtime calls them in exactly that order.
    """

    topology: str
    traffic_class: int
    world_size: int
    rank: int
    hca_names: tuple[str, ...]

    def local_blob(self) -> bytes: ...
    def connect(self, blobs: list[bytes]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def failed(self) -> bool: ...
    def error(self) -> str: ...
    def stats(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


__all__ = ["Geometry", "Transport", "TransportCapability"]
