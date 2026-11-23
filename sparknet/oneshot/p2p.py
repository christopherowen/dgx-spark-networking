"""One-shot RoCE point-to-point (PP P2P) runtime for multi-node pipeline parallelism.

Designed for DGX Spark clusters (integrated GB10 GPU with unified memory plus
ConnectX-7 200 GbE RoCEv2 NICs). Each rank owns one pinned host region shared
with its P2P proxy thread (``_roce_proxy.c``), partitioned into independent
per-peer directional 4-slot credit-flow-controlled rings:

* ``send(tensor, dst)`` stages into ``send[dst][seq & 3]`` (waiting on the
  receiver's RDMA-written credit only when all 4 slots are full), rings the
  proxy doorbell, and returns without waiting for the receiver.
* ``recv(out, src)`` waits for all stripe flags of ``seq`` (bounded by the
  fail-stop timeout in the control record), copies from ``recv[src][seq & 3]``
  with system-scope loads, and rings the ACK doorbell so the proxy RDMA-writes
  the updated credit back to the sender on lane 0 (strict RC QP FIFO order).
* Messages larger than ``slot_bytes`` stream in pipelined chunks inside one
  kernel launch with up to four slots in flight on the wire.
* ``broadcast(tensor, src)`` fans out over the same P2P channels.
* ``send_tensors`` / ``recv_tensors`` coalesce small multi-tensor payloads
  (e.g. pipeline decode activations) into a single flat RDMA transfer while
  streaming large prefill tensors zero-copy without extra device allocation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ._kernels import PACK_BYTES, P2PLaunch, kernel_setting_problems, p2p_recv_launcher, p2p_send_launcher
from ._proxy import P2PLayout, P2PProxy, load as _load_proxy_library
from .runtime import (
    API_VERSION,
    CTRL_TIMEOUT_WORD,
    CTRL_WAITED_WORD,
    DEFAULT_BLOCKS,
    DEFAULT_THREADS,
    MAX_LOCAL_HCAS,
    MAX_STRIPES,
    SUPPORTED_WORLD_SIZES,
    _SLOT_ALIGNMENT,
    _align_up,
    _capture_id,
    _checked_timeout_ms,
    _env_int,
    _exchange,
    _gid_networks,
    _grid_blocks,
    _hca_names,
    _normalize_device,
    _nullcontext,
    _removed_setting_problems,
    _subnets_from_env,
    discover_hcas,
)

logger = logging.getLogger(__name__)

DEFAULT_P2P_SLOT_BYTES = 1 * 1024 * 1024
DEFAULT_P2P_PACK_BYTES = 16 * 1024 * 1024
DEFAULT_P2P_TIMEOUT_MS = 30_000
CTRL_ERROR_KIND_WORD = 9

ENV_PP_HCA = "SPARKNET_ROCE_PP_HCA"
ENV_PP_PEER_HCAS = ("SPARKNET_ROCE_PP_PEER_HCAS",)
ENV_PP_TIMEOUT_MS = ("SPARKNET_ROCE_PP_TIMEOUT_MS", "SPARKNET_ROCE_TIMEOUT_MS")


def _pp_peer_hca_names_from_env() -> Optional[dict[int, tuple[str, ...]]]:
    """Parse ``SPARKNET_ROCE_PP_PEER_HCAS``: JSON ``{pp_peer_rank: [hca, ...]}``."""
    name = next((n for n in ENV_PP_PEER_HCAS if os.getenv(n)), None)
    if name is None:
        return None
    raw = os.environ[name]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    routes: dict[int, tuple[str, ...]] = {}
    for peer, names in value.items():
        try:
            peer_rank = int(peer)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} rank {peer!r} is not an integer") from exc
        if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
            raise ValueError(f"{name}[{peer!r}] must be a non-empty string list")
        routes[peer_rank] = tuple(names)
    return routes


def _resolve_pp_hca_topology(
    *,
    world_size: int,
    rank: int,
    hca_names: Optional[Sequence[str]],
    peer_hca_names: Optional[Mapping[int, Sequence[str]]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], int]:
    """Return opened HCAs, per-peer local routes, and stripe count for a P2P group.

    Explicit ``peer_hca_names`` or ``SPARKNET_ROCE_PP_PEER_HCAS`` maps every
    non-self rank in the PP group to its local stripe HCAs (e.g. the PP cable
    on ``oneshot-pp2-ring4``). Otherwise ``hca_names``, ``SPARKNET_ROCE_PP_HCA``,
    ``SPARKNET_ROCE_HCA``, or active RoCE discovery selects the shared rails
    (switched fabrics).
    """
    explicit_routes = peer_hca_names if peer_hca_names is not None else _pp_peer_hca_names_from_env()
    if explicit_routes is None:
        if hca_names:
            names = tuple(hca_names)[:MAX_STRIPES]
        else:
            names = _hca_names(ENV_PP_HCA) or discover_hcas()
            names = names[:MAX_STRIPES]
        if not names:
            raise RuntimeError("no active RDMA device found for RoCE P2P")
        routes = tuple(() if peer == rank else names for peer in range(world_size))
        return names, routes, len(names)

    normalized: dict[int, tuple[str, ...]] = {
        int(peer): tuple(names) for peer, names in explicit_routes.items()
    }
    expected_peers = set(range(world_size)) - {rank}
    if set(normalized) != expected_peers:
        missing = sorted(expected_peers - set(normalized))
        extra = sorted(set(normalized) - expected_peers)
        raise ValueError(
            f"pp peer_hca_names must name every non-self rank; missing={missing}, extra={extra}"
        )
    widths = {len(route) for route in normalized.values()}
    if len(widths) != 1 or not widths or not 1 <= next(iter(widths)) <= MAX_STRIPES:
        raise ValueError(f"every P2P peer HCA route must use the same 1..{MAX_STRIPES} stripe count")
    for peer, route in normalized.items():
        if len(set(route)) != len(route) or any(not name for name in route):
            raise ValueError(f"P2P peer {peer} HCA route contains an empty or duplicate name")
    routed_names = tuple(dict.fromkeys(name for peer in sorted(normalized) for name in normalized[peer]))
    names = tuple(hca_names) if hca_names else routed_names
    if len(set(names)) != len(names) or not 1 <= len(names) <= MAX_LOCAL_HCAS:
        raise ValueError(f"RoCE P2P requires 1..{MAX_LOCAL_HCAS} unique local HCA names")
    unknown = sorted(set(routed_names) - set(names))
    if unknown:
        raise ValueError(f"P2P peer HCA routes use unopened devices: {unknown}")
    routes = tuple(normalized.get(peer, ()) for peer in range(world_size))
    return names, routes, next(iter(widths))


def _peer_slot(rank: int, peer: int) -> int:
    return peer if peer < rank else peer - 1


class RoceOneshotP2P:
    """One-shot RDMA point-to-point runtime over the DGX Spark 200 GbE fabric."""

    algorithm = "oneshot-p2p"

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        slot_bytes: int = DEFAULT_P2P_SLOT_BYTES,
        pack_bytes: int = DEFAULT_P2P_PACK_BYTES,
        hca_names: Optional[Sequence[str]] = None,
        peer_hca_names: Optional[Mapping[int, Sequence[str]]] = None,
        subnets: Optional[Mapping[str, Any]] = None,
        threads: int = DEFAULT_THREADS,
        blocks: int = DEFAULT_BLOCKS,
        transport: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.device = _normalize_device(device)
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self._group = exchange_group
        self._closed = False
        self._lock = threading.Lock()
        self._proxy: Optional[Any] = None
        self._send_launcher: Optional[Callable[[P2PLaunch], None]] = None
        self._recv_launcher: Optional[Callable[[P2PLaunch], None]] = None
        self._align_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._pack_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._stream_event = torch.cuda.Event()
        self._last_stream: Optional[torch.cuda.Stream] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._capture_id = 0

        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"unsupported RoCE P2P world size {self.world_size}; supported sizes are {SUPPORTED_WORLD_SIZES}"
            )
        if int(slot_bytes) < _SLOT_ALIGNMENT:
            raise ValueError(f"slot_bytes must be at least {_SLOT_ALIGNMENT}")
        if int(pack_bytes) < 0:
            raise ValueError("pack_bytes must be non-negative")
        if int(threads) % 32 != 0 or int(threads) < 32 or int(threads) > 1024:
            raise ValueError("threads must be a multiple of 32 between 32 and 1024")
        if int(blocks) < 1 or int(blocks) & (int(blocks) - 1) != 0:
            raise ValueError("blocks must be a positive power of two")
        properties = torch.cuda.get_device_properties(self.device)
        resident = properties.multi_processor_count * (properties.max_threads_per_multi_processor // int(threads))
        if int(blocks) > resident:
            raise ValueError(
                f"blocks must fit on the device at once: {int(blocks)} blocks of {int(threads)} "
                f"threads, room for {resident}"
            )

        removed = _removed_setting_problems(os.environ) + kernel_setting_problems(os.environ)
        if removed:
            raise ValueError("; ".join(removed))

        self._threads = int(threads)
        self._blocks = int(blocks)
        self._counter_classes = self._blocks.bit_length()
        self._n_peers = self.world_size - 1
        self.timeout_ms = _checked_timeout_ms(_env_int(*ENV_PP_TIMEOUT_MS, default=DEFAULT_P2P_TIMEOUT_MS))
        self._transport_factory = transport if transport is not None else P2PProxy

        self.hca_names, self.peer_hca_names, self.stripe_count = _resolve_pp_hca_topology(
            world_size=self.world_size,
            rank=self.rank,
            hca_names=hca_names,
            peer_hca_names=peer_hca_names,
        )
        hca_indices = {name: index for index, name in enumerate(self.hca_names)}
        self._peer_hca_indices = tuple(
            tuple(hca_indices[name] for name in route) for route in self.peer_hca_names
        )
        raw_subnets = subnets if subnets is not None else _subnets_from_env()
        self.gid_networks = _gid_networks(self.hca_names, raw_subnets)

        aligned_slot = _align_up(int(slot_bytes), _SLOT_ALIGNMENT)
        self._layout = P2PLayout(self.world_size, aligned_slot)
        self._slot_bytes = aligned_slot
        self.slot_bytes = aligned_slot
        self.pack_bytes = _align_up(int(pack_bytes), PACK_BYTES) if int(pack_bytes) > 0 else 0

        with torch.cuda.device(self.device):
            self._region = torch.zeros(self._layout.total_bytes, dtype=torch.uint8, pin_memory=True)
            # Counter layout (int32 words on GPU):
            #   [0 .. n_peers - 1]                  : tx_epoch[pi]
            #   [n_peers .. 2*n_peers - 1]          : rx_epoch[pi]
            #   [2*n_peers .. 3*n_peers - 1]        : credit_ready[pi]
            #   [3*n_peers .. 4*n_peers - 1]        : stage_done[pi]
            #   [4*n_peers .. 5*n_peers - 1]        : arrival[pi]
            #   [5*n_peers .. 6*n_peers - 1]        : chunk_done[pi]
            #   [6*n_peers .. 6*n_peers + cc - 1]   : tx stage_counter per grid class
            #   [6*n_peers + cc .. + 2*cc - 1]      : rx ack_counter per grid class
            self._counters = torch.zeros(
                6 * self._n_peers + 2 * self._counter_classes,
                dtype=torch.int32,
                device=self.device,
            )

        host_ptr = self._region.data_ptr()
        device_ptr = self._device_pointer(host_ptr)
        if device_ptr != host_ptr:
            raise RuntimeError(
                "RoCE P2P needs host pointers that are directly device accessible "
                "(integrated GPU with unified addressing)"
            )
        self._ctrl_words = self._region[
            self._layout.ctrl_off : self._layout.ctrl_off + 4 * (CTRL_ERROR_KIND_WORD + 1)
        ].view(torch.int32)
        self._ctrl_np = self._ctrl_words.numpy()
        self._ctrl_np[CTRL_TIMEOUT_WORD] = 1000 * self.timeout_ms

        error: Optional[str] = None
        try:
            self._proxy = self._transport_factory(
                world_size=self.world_size,
                rank=self.rank,
                hca_names=self.hca_names,
                peer_hca_indices=self._peer_hca_indices,
                stripe_count=self.stripe_count,
                gid_networks=self.gid_networks,
                region_ptr=host_ptr,
                region_bytes=self._layout.total_bytes,
                slot_bytes=self._slot_bytes,
            )
            blob = self._proxy.local_blob()
        except Exception as exc:  # noqa: BLE001 - reported collectively below
            error = str(exc)
            blob = b""

        config = {
            "api_version": API_VERSION,
            "p2p_proxy_abi": _load_proxy_library().roce_p2p_abi_version() if error is None else None,
            "world_size": self.world_size,
            "hca_count": self.stripe_count,
            "open_hca_count": len(self.hca_names),
            "peer_stripes": tuple(len(route) for route in self.peer_hca_names),
            "traffic_class": self._proxy.traffic_class if error is None else None,
            "slot_bytes": self._slot_bytes,
            "pack_bytes": self.pack_bytes,
            "slots": self._layout.slots,
            "flag_stride": self._layout.flag_stride,
            "timeout_ms": self.timeout_ms,
            "threads": self._threads,
            "blocks": self._blocks,
        }
        statuses = _exchange((error, blob, config), exchange_group)
        failures = [f"rank {i}: {s[0]}" for i, s in enumerate(statuses) if s[0] is not None]
        if not failures:
            reference = statuses[0][2]
            rank_local_keys = {"open_hca_count", "peer_stripes"}
            for i, s in enumerate(statuses):
                differing = {
                    k: (reference[k], s[2].get(k))
                    for k in reference
                    if k not in rank_local_keys and s[2].get(k) != reference[k]
                }
                if differing:
                    failures.append(f"rank {i} P2P configuration differs from rank 0: {differing}")
            for i, s in enumerate(statuses):
                peer_stripes = tuple(s[2].get("peer_stripes", ()))
                expected = tuple(0 if peer == i else reference["hca_count"] for peer in range(self.world_size))
                if peer_stripes != expected:
                    failures.append(
                        f"rank {i} P2P peer stripe routes differ: expected {expected}, got {peer_stripes}"
                    )
        if failures:
            self.close()
            raise RuntimeError("RoCE P2P setup failed: " + "; ".join(failures))
        try:
            self._proxy.connect([s[1] for s in statuses])
            self._proxy.start()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        verdicts = _exchange(error, exchange_group)
        failures = [f"rank {i}: {v}" for i, v in enumerate(verdicts) if v is not None]
        if failures:
            self.close()
            raise RuntimeError("RoCE P2P connect failed: " + "; ".join(failures))
        if self.rank == 0:
            logger.info(
                "one-shot P2P ready: world=%d hcas=%s stripes=%d slot_bytes=%d pack_bytes=%d traffic_class=%d",
                self.world_size,
                ",".join(self.hca_names),
                self.stripe_count,
                self._slot_bytes,
                self.pack_bytes,
                self._proxy.traffic_class,
            )

    @staticmethod
    def _device_pointer(host_ptr: int) -> int:
        from cuda.bindings import runtime as cudart

        err, ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        return int(ptr)

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        slot_bytes: int = DEFAULT_P2P_SLOT_BYTES,
        pack_bytes: int = DEFAULT_P2P_PACK_BYTES,
        hca_names: Optional[Sequence[str]] = None,
        peer_hca_names: Optional[Mapping[int, Sequence[str]]] = None,
        threads: int = DEFAULT_THREADS,
        blocks: int = DEFAULT_BLOCKS,
    ) -> "RoceOneshotP2P":
        return cls(
            exchange_group=exchange_group,
            device=device,
            slot_bytes=slot_bytes,
            pack_bytes=pack_bytes,
            hca_names=hca_names,
            peer_hca_names=peer_hca_names,
            threads=threads,
            blocks=blocks,
        )

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        slot_bytes: int = DEFAULT_P2P_SLOT_BYTES,
        pack_bytes: int = DEFAULT_P2P_PACK_BYTES,
    ) -> "RoceOneshotP2P":
        return cls(
            exchange_group=process_group,
            device=device,
            slot_bytes=slot_bytes,
            pack_bytes=pack_bytes,
        )

    # -- policy & preparation ----------------------------------------------------

    def should_p2p(self, tensor: torch.Tensor) -> bool:
        """Rank-invariant eligibility: non-empty contiguous CUDA tensor (non-complex, non-sparse)."""
        if self._closed or self._proxy is None:
            raise RuntimeError("RoCE P2P runtime is closed")
        if not tensor.is_cuda or tensor.device != self.device or not tensor.is_contiguous():
            return False
        if tensor.is_complex() or tensor.is_sparse:
            return False
        return tensor.numel() > 0

    def _launcher_key(self) -> tuple[object, ...]:
        return (
            self.world_size,
            self.rank,
            self._threads,
            self._layout.slots,
            self._layout.flag_stride,
            self.stripe_count,
            self.device.index,
        )

    def _resolve_send_launcher(self, capturing: bool) -> Callable[[P2PLaunch], None]:
        if self._send_launcher is None:
            if capturing:
                raise RuntimeError("RoCE P2P send launcher was not prepared before CUDA graph capture; call prepare()")
            self._send_launcher = p2p_send_launcher(*self._launcher_key())
        return self._send_launcher

    def _resolve_recv_launcher(self, capturing: bool) -> Callable[[P2PLaunch], None]:
        if self._recv_launcher is None:
            if capturing:
                raise RuntimeError("RoCE P2P recv launcher was not prepared before CUDA graph capture; call prepare()")
            self._recv_launcher = p2p_recv_launcher(*self._launcher_key())
        return self._recv_launcher

    def _ensure_align_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._align_buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("RoCE P2P alignment scratch must be allocated before CUDA graph capture; call prepare()")
            self._align_buffers = (
                torch.empty(self._slot_bytes, dtype=torch.uint8, device=self.device),
                torch.empty(self._slot_bytes, dtype=torch.uint8, device=self.device),
            )
        return self._align_buffers

    def _ensure_pack_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pack_buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("RoCE P2P pack scratch must be allocated before CUDA graph capture; call prepare()")
            cap = max(self.pack_bytes, PACK_BYTES)
            self._pack_buffers = (
                torch.empty(cap, dtype=torch.uint8, device=self.device),
                torch.empty(cap, dtype=torch.uint8, device=self.device),
            )
        return self._pack_buffers

    def prepare(self) -> None:
        """Compile both P2P TileLang launchers and pre-allocate scratch ahead of capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("RoCE P2P preparation is refused inside a CUDA graph capture")
        with torch.cuda.device(self.device):
            self._resolve_send_launcher(capturing=False)
            self._resolve_recv_launcher(capturing=False)
            self._ensure_align_buffers()
            if self.pack_bytes > 0:
                self._ensure_pack_buffers()

    # -- stream ordering ---------------------------------------------------------

    def _order_stream(self, capturing: bool) -> None:
        current = torch.cuda.current_stream(self.device)
        if capturing:
            capture_id = _capture_id(current)
            if capture_id != self._capture_id:
                self._capture_id = capture_id
                self._capture_stream = current
            elif current != self._capture_stream:
                raise RuntimeError("RoCE P2P operations must all be captured on one stream")
            return
        if self._last_stream is not None and current != self._last_stream:
            current.wait_event(self._stream_event)

    def _mark_stream(self, capturing: bool) -> None:
        if capturing:
            return
        current = torch.cuda.current_stream(self.device)
        self._stream_event.record(current)
        self._last_stream = current

    # -- low-level aligned launch ------------------------------------------------

    def _check_peer(self, peer: int) -> int:
        peer = int(peer)
        if peer < 0 or peer >= self.world_size or peer == self.rank:
            raise ValueError(f"invalid P2P peer rank {peer} for rank {self.rank} of {self.world_size}")
        return peer

    def _counter_indices_tx(self, peer_slot: int, blocks: int) -> tuple[int, int, int, int]:
        n = self._n_peers
        cc_idx = int(blocks).bit_length() - 1
        return peer_slot, 6 * n + cc_idx, 2 * n + peer_slot, 3 * n + peer_slot

    def _counter_indices_rx(self, peer_slot: int, blocks: int) -> tuple[int, int, int, int]:
        n = self._n_peers
        cc_idx = int(blocks).bit_length() - 1
        return n + peer_slot, 6 * n + self._counter_classes + cc_idx, 4 * n + peer_slot, 5 * n + peer_slot

    def _launch_send_aligned(self, buf_u8: torch.Tensor, dst: int, capturing: bool) -> None:
        nbytes = buf_u8.numel()
        assert nbytes > 0 and nbytes % PACK_BYTES == 0 and buf_u8.data_ptr() % PACK_BYTES == 0
        total_packs = nbytes // PACK_BYTES
        chunk_packs = self._slot_bytes // PACK_BYTES
        num_chunks = (total_packs + chunk_packs - 1) // chunk_packs
        grid_x = _grid_blocks(min(total_packs, chunk_packs), self._threads, self._blocks)
        pi = _peer_slot(self.rank, dst)
        epoch_i, stage_i, ready_i, done_i = self._counter_indices_tx(pi, grid_x)
        launcher = self._resolve_send_launcher(capturing)
        launcher(
            P2PLaunch(
                tensor=buf_u8.view(torch.int32),
                total_packs=total_packs,
                chunk_packs=chunk_packs,
                num_chunks=num_chunks,
                peer_slot=pi,
                peer_rank=dst,
                region=self._region,
                data_off=self._layout.send_off,
                signal_off=self._layout.tx_ack_off,
                doorbell_off=self._layout.tx_ctrl_off,
                ctrl_off=self._layout.ctrl_off,
                slot_bytes=self._slot_bytes,
                counters=self._counters,
                epoch_index=epoch_i,
                barrier_index=stage_i,
                ready_index=ready_i,
                done_index=done_i,
                grid_x=grid_x,
            )
        )

    def _launch_recv_aligned(self, buf_u8: torch.Tensor, src: int, capturing: bool) -> None:
        nbytes = buf_u8.numel()
        assert nbytes > 0 and nbytes % PACK_BYTES == 0 and buf_u8.data_ptr() % PACK_BYTES == 0
        total_packs = nbytes // PACK_BYTES
        chunk_packs = self._slot_bytes // PACK_BYTES
        num_chunks = (total_packs + chunk_packs - 1) // chunk_packs
        grid_x = _grid_blocks(min(total_packs, chunk_packs), self._threads, self._blocks)
        pi = _peer_slot(self.rank, src)
        epoch_i, ack_i, arrival_i, done_i = self._counter_indices_rx(pi, grid_x)
        launcher = self._resolve_recv_launcher(capturing)
        launcher(
            P2PLaunch(
                tensor=buf_u8.view(torch.int32),
                total_packs=total_packs,
                chunk_packs=chunk_packs,
                num_chunks=num_chunks,
                peer_slot=pi,
                peer_rank=src,
                region=self._region,
                data_off=self._layout.recv_off,
                signal_off=self._layout.flag_off,
                doorbell_off=self._layout.ack_ctrl_off,
                ctrl_off=self._layout.ctrl_off,
                slot_bytes=self._slot_bytes,
                counters=self._counters,
                epoch_index=epoch_i,
                barrier_index=ack_i,
                ready_index=arrival_i,
                done_index=done_i,
                grid_x=grid_x,
            )
        )

    def _send_bytes(self, flat_u8: torch.Tensor, dst: int, capturing: bool) -> None:
        nbytes = flat_u8.numel()
        prefix = (nbytes // PACK_BYTES) * PACK_BYTES
        rem = nbytes - prefix
        if prefix > 0:
            if flat_u8.data_ptr() % PACK_BYTES == 0:
                self._launch_send_aligned(flat_u8[:prefix], dst, capturing)
            else:
                scratch, _ = self._ensure_align_buffers()
                offset = 0
                while offset < prefix:
                    step = min(self._slot_bytes, prefix - offset)
                    scratch[:step].copy_(flat_u8[offset : offset + step])
                    self._launch_send_aligned(scratch[:step], dst, capturing)
                    offset += step
        if rem > 0:
            scratch, _ = self._ensure_align_buffers()
            scratch[:rem].copy_(flat_u8[prefix:nbytes])
            self._launch_send_aligned(scratch[:PACK_BYTES], dst, capturing)

    def _recv_bytes(self, flat_u8: torch.Tensor, src: int, capturing: bool) -> None:
        nbytes = flat_u8.numel()
        prefix = (nbytes // PACK_BYTES) * PACK_BYTES
        rem = nbytes - prefix
        if prefix > 0:
            if flat_u8.data_ptr() % PACK_BYTES == 0:
                self._launch_recv_aligned(flat_u8[:prefix], src, capturing)
            else:
                _, scratch = self._ensure_align_buffers()
                offset = 0
                while offset < prefix:
                    step = min(self._slot_bytes, prefix - offset)
                    self._launch_recv_aligned(scratch[:step], src, capturing)
                    flat_u8[offset : offset + step].copy_(scratch[:step])
                    offset += step
        if rem > 0:
            _, scratch = self._ensure_align_buffers()
            self._launch_recv_aligned(scratch[:PACK_BYTES], src, capturing)
            flat_u8[prefix:nbytes].copy_(scratch[:rem])

    # -- public operations -------------------------------------------------------

    def send(self, tensor: torch.Tensor, dst: int, *, stream: object = None) -> None:
        """Send contiguous CUDA ``tensor`` asynchronously to peer rank ``dst``."""
        dst = self._check_peer(dst)
        with self._lock:
            self.check_health()
            if not self.should_p2p(tensor):
                raise ValueError("tensor is not eligible for RoCE P2P send")
            context = torch.cuda.stream(stream) if stream is not None else _nullcontext()
            with torch.cuda.device(self.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                self._order_stream(capturing)
                flat_u8 = tensor.reshape(-1).view(torch.uint8)
                self._send_bytes(flat_u8, dst, capturing)
                self._mark_stream(capturing)
        if not capturing:
            self.check_health()

    def recv(self, out: torch.Tensor, src: int, *, stream: object = None) -> torch.Tensor:
        """Receive contiguous CUDA tensor from peer rank ``src`` into ``out``."""
        src = self._check_peer(src)
        with self._lock:
            self.check_health()
            if not self.should_p2p(out):
                raise ValueError("out is not eligible for RoCE P2P recv")
            context = torch.cuda.stream(stream) if stream is not None else _nullcontext()
            with torch.cuda.device(self.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                self._order_stream(capturing)
                flat_u8 = out.reshape(-1).view(torch.uint8)
                self._recv_bytes(flat_u8, src, capturing)
                self._mark_stream(capturing)
        if not capturing:
            self.check_health()
        return out

    def broadcast(self, tensor: torch.Tensor, src: int, *, stream: object = None) -> torch.Tensor:
        """Broadcast ``tensor`` from rank ``src`` to every other rank in the P2P group."""
        src = int(src)
        if src < 0 or src >= self.world_size:
            raise ValueError(f"invalid broadcast source rank {src} for world_size {self.world_size}")
        if self.rank == src:
            for peer in range(self.world_size):
                if peer != src:
                    self.send(tensor, dst=peer, stream=stream)
        else:
            self.recv(tensor, src=src, stream=stream)
        return tensor

    def send_tensors(self, tensors: Sequence[torch.Tensor], dst: int, *, stream: object = None) -> None:
        """Send a sequence of contiguous CUDA tensors to ``dst``.

        When ``len(tensors) == 1`` sends directly zero-copy. When ``len(tensors) > 1``
        and the 16-byte-aligned packed size fits in ``pack_bytes``, packs into a
        single flat RDMA transfer; otherwise streams each tensor zero-copy.
        """
        if not tensors:
            return
        if len(tensors) == 1:
            self.send(tensors[0], dst, stream=stream)
            return
        dst = self._check_peer(dst)
        with self._lock:
            self.check_health()
            total = 0
            slices: list[tuple[int, int, torch.Tensor]] = []
            for t in tensors:
                if not self.should_p2p(t):
                    raise ValueError("tensor is not eligible for RoCE P2P send_tensors")
                n = t.numel() * t.element_size()
                slices.append((total, total + n, t))
                total = _align_up(total + n, PACK_BYTES)
            context = torch.cuda.stream(stream) if stream is not None else _nullcontext()
            with torch.cuda.device(self.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                self._order_stream(capturing)
                if 0 < total <= self.pack_bytes:
                    pack_send, _ = self._ensure_pack_buffers()
                    flat = pack_send[:total]
                    for start, stop, t in slices:
                        flat[start:stop].copy_(t.reshape(-1).view(torch.uint8))
                    self._launch_send_aligned(flat, dst, capturing)
                else:
                    for _, _, t in slices:
                        self._send_bytes(t.reshape(-1).view(torch.uint8), dst, capturing)
                self._mark_stream(capturing)
        if not capturing:
            self.check_health()

    def recv_tensors(self, outs: Sequence[torch.Tensor], src: int, *, stream: object = None) -> list[torch.Tensor]:
        """Receive a matching sequence of contiguous CUDA tensors from ``src``."""
        if not outs:
            return []
        if len(outs) == 1:
            self.recv(outs[0], src, stream=stream)
            return [outs[0]]
        src = self._check_peer(src)
        with self._lock:
            self.check_health()
            total = 0
            slices: list[tuple[int, int, torch.Tensor]] = []
            for t in outs:
                if not self.should_p2p(t):
                    raise ValueError("out is not eligible for RoCE P2P recv_tensors")
                n = t.numel() * t.element_size()
                slices.append((total, total + n, t))
                total = _align_up(total + n, PACK_BYTES)
            context = torch.cuda.stream(stream) if stream is not None else _nullcontext()
            with torch.cuda.device(self.device), context:
                capturing = torch.cuda.is_current_stream_capturing()
                self._order_stream(capturing)
                if 0 < total <= self.pack_bytes:
                    _, pack_recv = self._ensure_pack_buffers()
                    flat = pack_recv[:total]
                    self._launch_recv_aligned(flat, src, capturing)
                    for start, stop, t in slices:
                        t.reshape(-1).view(torch.uint8).copy_(flat[start:stop])
                else:
                    for _, _, t in slices:
                        self._recv_bytes(t.reshape(-1).view(torch.uint8), src, capturing)
                self._mark_stream(capturing)
        if not capturing:
            self.check_health()
        return list(outs)

    @contextmanager
    def capture(self, stream: object = None):
        """Context manager for CUDA graph capture; launchers must already be prepared."""
        self._last_stream = None
        self._capture_stream = None
        self._capture_id = 0
        try:
            yield self
        finally:
            self._capture_stream = None
            self._capture_id = 0
            self._last_stream = None

    # -- fail-stop & diagnostics -------------------------------------------------

    def set_timeout_ms(self, timeout_ms: int) -> None:
        self.timeout_ms = _checked_timeout_ms(timeout_ms)
        self._ctrl_np[CTRL_TIMEOUT_WORD] = 1000 * self.timeout_ms

    def failure(self) -> Optional[str]:
        parts = []
        failed_seq = int(self._ctrl_np[2])
        if failed_seq != 0:
            waited_s = int(self._ctrl_np[CTRL_WAITED_WORD]) / 1e6
            kind = int(self._ctrl_np[CTRL_ERROR_KIND_WORD])
            if kind == 1:
                parts.append(
                    f"RoCE P2P send on rank {self.rank} waited {waited_s:.3f} s for credit from rank "
                    f"{int(self._ctrl_np[3])} (timeout {self.timeout_ms / 1000:g} s) at sequence {failed_seq}; "
                    "the GPU was stopped there"
                )
            else:
                parts.append(
                    f"RoCE P2P recv on rank {self.rank} saw nothing from rank {int(self._ctrl_np[3])}, "
                    f"HCA {int(self._ctrl_np[6])}, for {waited_s:.3f} s "
                    f"(timeout {self.timeout_ms / 1000:g} s) at sequence {failed_seq}; "
                    "the GPU was stopped there"
                )
        if self._proxy is not None and self._proxy.failed():
            parts.append(f"RoCE P2P proxy failed: {self._proxy.error()}")
        return "; ".join(parts) if parts else None

    def check_health(self) -> None:
        report = self.failure()
        if report is not None:
            raise RuntimeError(report)

    @property
    def poisoned(self) -> bool:
        return self.failure() is not None

    def stats(self) -> dict[str, Any]:
        n = self._n_peers
        info: dict[str, Any] = {
            "world_size": self.world_size,
            "rank": self.rank,
            "hcas": list(self.hca_names),
            "peer_hca_routes": {
                str(peer): list(route)
                for peer, route in enumerate(self.peer_hca_names)
                if peer != self.rank
            },
            "gid_subnets": {name: str(net) for name, net in zip(self.hca_names, self.gid_networks, strict=True)},
            "gid_indices": (
                {name: idx for name, idx in zip(self.hca_names, self._proxy.gid_indices, strict=True)}
                if self._proxy is not None
                else {}
            ),
            "stripe_count": self.stripe_count,
            "slot_bytes": self._slot_bytes,
            "pack_bytes": self.pack_bytes,
            "slots": self._layout.slots,
            "tx_epochs": [int(v) for v in self._counters[:n].tolist()],
            "rx_epochs": [int(v) for v in self._counters[n : 2 * n].tolist()],
            "error_seq": int(self._ctrl_words[2].item()),
            "error_peer": int(self._ctrl_words[3].item()),
            "error_lane": int(self._ctrl_words[6].item()),
            "error_kind": int(self._ctrl_words[CTRL_ERROR_KIND_WORD].item()),
            "error_waited_us": int(self._ctrl_words[CTRL_WAITED_WORD].item()),
            "timeout_ms": self.timeout_ms,
            "threads": self._threads,
            "blocks": self._blocks,
        }
        if self._proxy is not None:
            info.update(self._proxy.stats())
        return info

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(self.device)
            if self._proxy is not None:
                self._proxy.close()
                self._proxy = None

    def __del__(self) -> None:  # pragma: no cover - defensive teardown
        with contextlib.suppress(Exception):
            self.close()


__all__ = [
    "DEFAULT_P2P_PACK_BYTES",
    "DEFAULT_P2P_SLOT_BYTES",
    "DEFAULT_P2P_TIMEOUT_MS",
    "RoceOneshotP2P",
    "_resolve_pp_hca_topology",
]
