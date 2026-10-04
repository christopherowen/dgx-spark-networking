"""One policy in execution and reporting.

A collective policy fixes, before any graph capture, which backend carries
each operation: the one-shot collectives for the small all-reduces and all-gathers on the
next-token path, NCCL for everything else (larger payloads, reduce-scatter,
variable collectives). The decision depends only on operation, dtype, shape,
contiguity and byte size, which tensor-parallel ranks share, never on pointer
values, so every rank routes the same call. Transport failure is a coordinated
failure, never a local switch to another backend.

Three limits are deliberately distinct:

- the all-reduce **dispatch** ceiling: the largest input one-shot carries in
  serving (the measured crossover with NCCL Ring is near 1 MiB on four nodes);
- the all-reduce **capacity**: the registered and primed slot size, which may
  exceed dispatch (vLLM's sequence-parallel prefill threshold reads it);
- the all-gather limit on the **input shard**: the output is world_size times
  larger.
"""

from __future__ import annotations

from dataclasses import dataclass

PACK_BYTES = 16
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")
_GATHER_REJECTED_DTYPES = ("bool", "complex64", "complex128")

ENV_CAPACITY = ("SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES",)
ENV_DISPATCH = ("SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES",)
ENV_GATHER = ("SPARKNET_ROCE_ALLGATHER_MAX_BYTES",)


@dataclass(frozen=True)
class CollectivePolicy:
    all_reduce_dispatch_bytes: int
    all_reduce_capacity_bytes: int
    all_gather_shard_bytes: int

    def problems(self) -> list[str]:
        errors = []
        for label, value in (("all_reduce_dispatch_bytes", self.all_reduce_dispatch_bytes),
                             ("all_reduce_capacity_bytes", self.all_reduce_capacity_bytes),
                             ("all_gather_shard_bytes", self.all_gather_shard_bytes)):
            if value < PACK_BYTES or value % PACK_BYTES:
                errors.append(f"{label} must be a multiple of {PACK_BYTES} bytes, at least {PACK_BYTES}")
        if self.all_reduce_dispatch_bytes > self.all_reduce_capacity_bytes:
            errors.append("all-reduce dispatch must not exceed registered capacity")
        return errors

    def all_reduce_backend(self, nbytes: int, dtype: str, *, contiguous: bool = True) -> str:
        """``oneshot`` for an eligible input, else ``nccl``."""
        if (contiguous and dtype in SUPPORTED_DTYPES and 0 < nbytes <= self.all_reduce_dispatch_bytes
                and nbytes % PACK_BYTES == 0):
            return "oneshot"
        return "nccl"

    def all_gather_backend(self, shard_bytes: int, dtype: str, *, dim: int, ndim: int,
                           contiguous: bool = True) -> str:
        """``oneshot`` for a contiguous shard concatenated along dim 0 or the last dim within the limit."""
        if dim < 0:
            dim += ndim
        if (contiguous and ndim > 0 and dtype not in _GATHER_REJECTED_DTYPES and dim in (0, ndim - 1)
                and 0 < shard_bytes <= self.all_gather_shard_bytes):
            return "oneshot"
        return "nccl"

    @staticmethod
    def reduce_scatter_backend() -> str:
        return "nccl"

    def environment(self) -> dict[str, str]:
        return {
            "SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES": str(self.all_reduce_capacity_bytes),
            "SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": str(self.all_reduce_dispatch_bytes),
            "SPARKNET_ROCE_ALLGATHER_MAX_BYTES": str(self.all_gather_shard_bytes),
        }

    @classmethod
    def from_environment(cls, env: dict[str, str]) -> "CollectivePolicy":
        """Read the limits, in bytes, as a recipe sets them."""

        def first(names: tuple[str, ...]) -> int | None:
            for name in names:
                if env.get(name):
                    return int(env[name])
            return None

        capacity = first(ENV_CAPACITY)
        gather = first(ENV_GATHER)
        if capacity is None or gather is None:
            raise ValueError("the environment names no one-shot all-reduce capacity or all-gather limit")
        dispatch = first(ENV_DISPATCH)
        policy = cls(dispatch if dispatch is not None else capacity, capacity, gather)
        errors = policy.problems()
        if errors:
            raise ValueError("; ".join(errors))
        return policy


TP3_POLICY = CollectivePolicy(2 * 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024)
TP4_POLICY = CollectivePolicy(1024 * 1024, 2 * 1024 * 1024, 2 * 1024 * 1024)


def policy_for_profile(name: str) -> CollectivePolicy | None:
    """The policy a named transport profile measured; None for an NCCL-only profile."""
    from sparknet.nccl.profiles import profile

    roce = profile(name)["oneshot"]
    if roce["ALLREDUCE_CAPACITY_BYTES"] is None:
        return None
    capacity = int(roce["ALLREDUCE_CAPACITY_BYTES"])
    dispatch = int(roce["ALLREDUCE_DISPATCH_MAX_BYTES"] or capacity)
    return CollectivePolicy(dispatch, capacity, int(roce["ALLGATHER_MAX_BYTES"]))


__all__ = ["PACK_BYTES", "SUPPORTED_DTYPES", "CollectivePolicy", "TP3_POLICY", "TP4_POLICY", "policy_for_profile"]
