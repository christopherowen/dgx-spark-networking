"""One-shot RoCE point-to-point adapter for vLLM pipeline parallelism on DGX Spark.

Mirrors ``SparknetOneShotAllReduce``'s fail-stop contract for the pipeline
process group (``get_pp_group()``):

- All pipeline ranks vote on capability and byte-size settings over the CPU
  (gloo) group before constructing ``sparknet.oneshot.P2P`` (no torch NCCL
  communicator is required for setup). Any rank unable to participate fails
  initialization on every rank.
- Dispatch is rank-invariant: eligibility depends only on dtype, shape,
  contiguity and device, never on pointer alignment.
- Failures are fail-stop, never a fallback: a timed-out credit or flag wait
  stops the GPU and records the missing peer, direction and elapsed time in
  pinned host memory; ``check_health`` raises on the host.
- Launchers and alignment/pack scratch are prepared at construction so no JIT
  or allocation happens inside a serving step or CUDA graph capture.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .roce_all_reduce import REQUIRED_API_VERSION, parse_byte_size

logger = logging.getLogger(__name__)

ENV_PP_SLOT_BYTES = ("SPARKNET_ROCE_PP_SLOT_BYTES",)
ENV_PP_PACK_BYTES = ("SPARKNET_ROCE_PP_PACK_BYTES",)


def _env(names: Sequence[str], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


class SparknetOneShotP2P:
    """Route pipeline-parallel point-to-point sends, recvs and broadcasts to ``sparknet.oneshot.P2P``."""

    backend_name = "SPARKNET_ONESHOT_P2P"

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        *,
        in_the_same_node: Sequence[bool] | None = None,
        slot_bytes: int | None = None,
        pack_bytes: int | None = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._runtime = None
        self._announced = False

        if self.world_size < 2:
            logger.info("one-shot P2P skipped: pipeline group has a single rank.")
            return
        if in_the_same_node is not None and all(in_the_same_node):
            logger.info("one-shot P2P skipped: pipeline group is single-node.")
            return

        reason, limits = self._local_capability(slot_bytes=slot_bytes, pack_bytes=pack_bytes)
        verdict = self._exchange_vote(reason, limits)
        if verdict is not None:
            raise RuntimeError(f"one-shot P2P policy unavailable: {verdict}")
        resolved_slot, resolved_pack = limits

        from sparknet import oneshot

        try:
            self._runtime = oneshot.P2P.from_exchange_group(
                exchange_group=group,
                device=device,
                slot_bytes=resolved_slot,
                pack_bytes=resolved_pack,
            )
            self._runtime.prepare()
        except Exception as exc:  # noqa: BLE001 - ranks coordinated by runtime
            raise RuntimeError("one-shot P2P initialization failed") from exc

        self.disabled = False
        if self.rank == 0:
            logger.info(
                "Using one-shot RoCE P2P: world=%d, hcas=%s, stripes=%d, "
                "slot_bytes=%d, pack_bytes=%d; backend failures are fatal.",
                self.world_size,
                ",".join(self._runtime.hca_names),
                self._runtime.stripe_count,
                self._runtime.slot_bytes,
                self._runtime.pack_bytes,
            )

    def _local_capability(
        self, *, slot_bytes: int | None, pack_bytes: int | None
    ) -> tuple[str | None, tuple[int, int] | None]:
        try:
            from sparknet import oneshot
        except ImportError as exc:
            return f"sparknet.oneshot is not importable: {exc}", None
        api = getattr(oneshot, "API_VERSION", None)
        if api != REQUIRED_API_VERSION:
            return f"sparknet.oneshot API version {api}, adapter needs {REQUIRED_API_VERSION}", None
        if not oneshot.is_supported(self.device):
            return "needs an integrated GPU with an active RDMA device", None
        try:
            resolved_slot = int(slot_bytes) if slot_bytes is not None else parse_byte_size(_env(ENV_PP_SLOT_BYTES, "1MB"))
            resolved_pack = int(pack_bytes) if pack_bytes is not None else parse_byte_size(_env(ENV_PP_PACK_BYTES, "16MB"))
        except Exception as exc:  # noqa: BLE001
            return f"invalid one-shot P2P size limit: {exc}", None
        return None, (resolved_slot, resolved_pack)

    def _exchange_vote(self, reason: str | None, limits: tuple[int, int] | None) -> str | None:
        votes: list[tuple[str | None, tuple[int, int] | None]] = [(None, None)] * self.world_size
        dist.all_gather_object(votes, (reason, limits), group=self.group)
        failures = [f"rank {i}: {r}" for i, (r, _) in enumerate(votes) if r]
        if failures:
            return "; ".join(failures)
        reference = votes[0][1]
        differing = [f"rank {i}: {lim}" for i, (_, lim) in enumerate(votes) if lim != reference]
        if differing:
            return f"P2P size limits differ across ranks (rank 0: {reference}; " + "; ".join(differing) + ")"
        return None

    # -- policy surface ----------------------------------------------------------

    @property
    def slot_bytes(self) -> int:
        return 0 if self.disabled else int(self._runtime.slot_bytes)

    @property
    def pack_bytes(self) -> int:
        return 0 if self.disabled else int(self._runtime.pack_bytes)

    def should_p2p(self, tensor: torch.Tensor) -> bool:
        return not self.disabled and self._runtime.should_p2p(tensor)

    def check_health(self) -> None:
        if not self.disabled and self._runtime is not None:
            self._runtime.check_health()

    def failure(self) -> str | None:
        if self.disabled or self._runtime is None:
            return None
        return self._runtime.failure()

    def _announce(self, label: str, nbytes: int) -> None:
        if not self._announced:
            self._announced = True
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "one-shot P2P is live: first %s is %d bytes (slot_bytes=%d, pack_bytes=%d).",
                label,
                nbytes,
                self.slot_bytes,
                self.pack_bytes,
            )

    def send(self, tensor: torch.Tensor, dst: int, *, stream: object = None) -> None:
        self._announce("send", tensor.numel() * tensor.element_size())
        self._runtime.send(tensor, dst=dst, stream=stream)

    def recv(self, out: torch.Tensor, src: int, *, stream: object = None) -> torch.Tensor:
        self._announce("recv", out.numel() * out.element_size())
        return self._runtime.recv(out, src=src, stream=stream)

    def broadcast(self, tensor: torch.Tensor, src: int, *, stream: object = None) -> torch.Tensor:
        self._announce("broadcast", tensor.numel() * tensor.element_size())
        return self._runtime.broadcast(tensor, src=src, stream=stream)

    def send_tensors(self, tensors: Sequence[torch.Tensor], dst: int, *, stream: object = None) -> None:
        total = sum(t.numel() * t.element_size() for t in tensors)
        self._announce("send_tensors", total)
        self._runtime.send_tensors(tensors, dst=dst, stream=stream)

    def recv_tensors(self, outs: Sequence[torch.Tensor], src: int, *, stream: object = None) -> list[torch.Tensor]:
        total = sum(t.numel() * t.element_size() for t in outs)
        self._announce("recv_tensors", total)
        return self._runtime.recv_tensors(outs, src=src, stream=stream)

    def send_tensor_dict_fixed(
        self,
        tensor_dict: Mapping[str, torch.Tensor],
        keys: Sequence[str],
        dst: int,
        *,
        stream: object = None,
    ) -> None:
        """Send a fixed-schema tensor dictionary without Gloo metadata round-trips."""
        self.send_tensors([tensor_dict[k] for k in keys], dst=dst, stream=stream)

    def recv_tensor_dict_fixed(
        self,
        out_dict: Mapping[str, torch.Tensor],
        keys: Sequence[str],
        src: int,
        *,
        stream: object = None,
    ) -> dict[str, torch.Tensor]:
        """Receive a fixed-schema tensor dictionary into pre-allocated ``out_dict``."""
        self.recv_tensors([out_dict[k] for k in keys], src=src, stream=stream)
        return dict(out_dict)

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled:
            yield
            return
        with self._runtime.capture(stream=stream):
            yield

    def stats(self) -> dict[str, object]:
        return {} if self.disabled or self._runtime is None else self._runtime.stats()

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True


__all__ = ["SparknetOneShotP2P"]
