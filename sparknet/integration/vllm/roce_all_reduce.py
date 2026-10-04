"""RoCEnante adapter for vLLM's CUDA device communicator (multi-node DGX Spark TP).

A thin shim: capability voting, construction, preparation and size gating live
here; the protocol lives in ``sparknet.rocenante``. It replaces the
``B12xRoceAllReduce`` class of the Local Inference Lab vLLM fork one for one
(same constructor keywords, same methods), and keeps the fail-stop policy of
that fork's explicit-collective-policy patch:

- Every rank parses the size limits and checks the API version before the
  vote; the parsed limits are exchanged and must be identical, and the
  runtime itself refuses ranks whose ABI, HCA count, slot geometry, spin
  limit or launch geometry differ. Any rank that cannot take part aborts
  initialization on every rank. Explicit multi-node selection is a policy,
  not a request to try other backends on failure.
- Dispatch is rank-invariant: eligibility depends on dtype, shape, contiguity
  and size, never on pointer values, so all ranks route the same collective.
- Failures are fail-stop, never a fallback: a wait that times out freezes the
  runtime, later launches do nothing, and ``check_health`` (called by the
  worker after each step's host synchronization) raises so the step's output
  never leaves the worker. Peers starve on the stalled rank and raise too.
- The runtime orders collectives across streams with an event and requires a
  single stream inside a CUDA graph capture, which is how vLLM captures.

The adapter prepares the launchers at construction (all ranks construct
together, before any graph capture), so no JIT happens inside a serving step.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

REQUIRED_API_VERSION = 1
ENV_ENABLE = ("SPARKNET_ENABLE_ROCE_ALLREDUCE", "VLLM_ENABLE_ROCE_ALLREDUCE")
ENV_ALLREDUCE_LIMIT = ("SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES", "VLLM_ROCE_ALLREDUCE_MAX_SIZE")
ENV_ALLGATHER_LIMIT = ("SPARKNET_ROCE_ALLGATHER_MAX_BYTES", "VLLM_ROCE_ALLGATHER_MAX_SIZE")


def parse_byte_size(value: str) -> int:
    """``2MB`` or ``2097152``; KB/MB/GB are binary units, as vLLM parses them."""
    match = re.fullmatch(r"\s*([1-9][0-9]*)\s*(B|KB|MB|GB)?\s*", str(value), re.I)
    if not match:
        raise ValueError(f"invalid byte size {value!r}")
    unit = (match[2] or "B").upper()
    return int(match[1]) * {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}[unit]


def _env(names: Sequence[str], default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def enabled() -> bool:
    return _env(ENV_ENABLE, "0") == "1"


class SparknetRoceAllReduce:
    """Route eligible tensor-parallel all-reduces and all-gathers to ``sparknet.rocenante``."""

    backend_name = "SPARKNET_ROCENANTE"

    def __init__(
        self,
        group: ProcessGroup,
        device_group: ProcessGroup | None,
        device: torch.device,
        *,
        global_ranks: Sequence[int] | None = None,
        nccl_available: bool = True,
        in_the_same_node: Sequence[bool] | None = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._runtime = None
        self._announced = False
        self._announced_gather = False
        self.global_ranks = tuple(int(r) for r in (global_ranks if global_ranks is not None else range(self.world_size)))
        if len(self.global_ranks) != self.world_size:
            raise ValueError("RoCE global ranks must match the process group")
        if device_group is None:
            raise RuntimeError("RoCEnante requires a CUDA process group.")
        if in_the_same_node is not None and all(in_the_same_node):
            logger.info("RoCEnante skipped: group is single-node.")
            return

        # Vote before the collective constructor so a rank that cannot take
        # part fails initialization on every rank instead of leaving peers in
        # the runtime's setup exchange. The parsed limits travel with the vote.
        reason, limits = self._local_capability()
        if not nccl_available:
            reason = "RoCEnante policy requires an available NCCL communicator"
        verdict = self._exchange_vote(reason, limits)
        if verdict is not None:
            raise RuntimeError(f"RoCEnante policy unavailable: {verdict}")
        max_size, max_gather = limits

        from sparknet import rocenante

        try:
            # Exchange setup over the CPU (gloo) group: using the torch NCCL
            # group would create a torch NCCL communicator that vLLM otherwise
            # never needs, costing about 3.4 GB of unified memory per rank.
            self._runtime = rocenante.AllReduce.from_exchange_group(
                exchange_group=group, device=device, max_size=max_size, max_gather_bytes=max_gather,
            )
            self._runtime.prepare((torch.float16, torch.bfloat16, torch.float32), padded_gather=True)
        except Exception as exc:  # noqa: BLE001 - the runtime already coordinated ranks
            raise RuntimeError("RoCEnante policy initialization failed") from exc
        self.disabled = False
        if self.rank == 0:
            logger.info(
                "Using RoCEnante (sparknet one-shot RoCE collectives): world=%d, topology=%s, hcas=%s, "
                "all-reduce dispatch <=%d bytes, registered all-reduce capacity=%d bytes, "
                "all-gather input shard <=%d bytes. NCCL handles ineligible inputs and "
                "reduce-scatter; backend failures are fatal.",
                self.world_size, self._runtime.topology, ",".join(self._runtime.hca_names),
                self.all_reduce_max_bytes, self.all_reduce_capacity_bytes, self.all_gather_max_bytes,
            )

    def _local_capability(self) -> tuple[str | None, tuple[int, int] | None]:
        try:
            from sparknet import rocenante
        except ImportError as exc:  # missing package or a broken native build
            return f"sparknet.rocenante is not importable: {exc}", None
        api = getattr(rocenante, "API_VERSION", None)
        if api != REQUIRED_API_VERSION:
            return f"sparknet.rocenante API version {api}, adapter needs {REQUIRED_API_VERSION}", None
        if not rocenante.is_supported(self.device):
            return "needs an integrated GPU with an active RDMA device", None
        try:
            limits = (parse_byte_size(_env(ENV_ALLREDUCE_LIMIT, "2MB")), parse_byte_size(_env(ENV_ALLGATHER_LIMIT, "4MB")))
        except Exception as exc:  # noqa: BLE001 - reported through the vote
            return f"invalid RoCEnante size limit: {exc}", None
        return None, limits

    def _exchange_vote(self, reason: str | None, limits: tuple[int, int] | None) -> str | None:
        votes: list[tuple[str | None, tuple[int, int] | None]] = [(None, None)] * self.world_size
        dist.all_gather_object(votes, (reason, limits), group=self.group)
        failures = [f"rank {i}: {r}" for i, (r, _) in enumerate(votes) if r]
        if failures:
            return "; ".join(failures)
        reference = votes[0][1]
        differing = [f"rank {i}: {lim}" for i, (_, lim) in enumerate(votes) if lim != reference]
        if differing:
            return f"size limits differ across ranks (rank 0: {reference}; " + "; ".join(differing) + ")"
        return None

    # -- policy surface ----------------------------------------------------------

    @property
    def all_reduce_max_bytes(self) -> int:
        """The largest message ``should_custom_ar`` accepts; 0 when disabled."""
        return 0 if self.disabled else int(self._runtime.dispatch_max_bytes)

    @property
    def all_reduce_capacity_bytes(self) -> int:
        """Registered/primed capacity, independent of the dispatch cutoff."""
        return 0 if self.disabled else int(self._runtime.max_size)

    @property
    def all_gather_max_bytes(self) -> int:
        """Largest eligible input shard, not the gathered output size."""
        return 0 if self.disabled else int(self._runtime.max_gather_bytes)

    def check_health(self) -> None:
        if not self.disabled and self._runtime is not None:
            self._runtime.check_health()

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._runtime.should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        if not self._announced:
            self._announced = True
            log = logger.info if self.rank == 0 else logger.debug
            log("RoCEnante all-reduce is live: first routed all-reduce is %d bytes (%s); dispatch limit=%d bytes, registered capacity=%d bytes.",
                inp.numel() * inp.element_size(), str(inp.dtype).replace("torch.", ""),
                self.all_reduce_max_bytes, self.all_reduce_capacity_bytes)
        return self._runtime.all_reduce(inp)

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return not self.disabled and self._runtime.should_all_gather(inp, dim)

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        if not self._announced_gather:
            self._announced_gather = True
            log = logger.info if self.rank == 0 else logger.debug
            log("RoCEnante all-gather is live: first routed shard is %s %s along dim %d; input shard dispatch limit=%d bytes.",
                tuple(inp.shape), str(inp.dtype).replace("torch.", ""), dim, self.all_gather_max_bytes)
        return self._runtime.all_gather(inp, dim=dim)

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled:
            yield
            return
        with self._runtime.capture(stream=stream):
            yield

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True


__all__ = ["REQUIRED_API_VERSION", "SparknetRoceAllReduce", "enabled", "parse_byte_size"]
