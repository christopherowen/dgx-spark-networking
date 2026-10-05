# Collective policy

One policy in execution and reporting, decided before graph capture and
identical on every rank.

| Operation and condition | Backend |
| --- | --- |
| All-reduce: contiguous CUDA FP16/BF16/FP32, size a positive multiple of 16 bytes, up to the dispatch limit | One-shot |
| Other all-reduce inputs | NCCL |
| All-gather: contiguous CUDA shard up to the shard limit, concatenated along dim 0 or the last dim, not bool, complex or sparse | One-shot (direct layout for 16-byte rows, padded scratch plus reshape otherwise) |
| Other all-gather inputs | NCCL |
| Reduce-scatter, variable collectives, broadcast, send/receive | NCCL |
| Lost communicator, transport exception, timed-out wait | error on every rank; no switch to another backend |

Three distinct limits (`sparknet.policy.CollectivePolicy`):

- **dispatch** (`all_reduce_dispatch_bytes`): the largest all-reduce
  one-shot carries in serving. The measured crossover with NCCL Ring on
  four nodes is near 1 to 1.25 MiB (one-shot sends three input payloads per
  rank; NCCL's ring 1.5), so the TP4 profile dispatches at 1 MiB.
- **capacity** (`all_reduce_capacity_bytes`): the registered and primed slot
  size, 2 MiB in both profiles. vLLM's sequence-parallel prefill threshold
  (205 tokens at 5,120 BF16 columns) is derived from it, so tuning dispatch
  does not move model scheduling.
- **all-gather shard** (`all_gather_shard_bytes`): the input shard per rank;
  the output is world_size times larger. 4 MiB on three nodes, 2 MiB on four
  (NCCL wins above it there).

Measured 2026-10-05 (`docs/oneshot.md`, Where the NCCL cut is): the ring's
all-reduce crossover is 1.25 to 1.5 MiB against unbalanced NCCL and lower
against the balanced policy, so the 1 MiB dispatch holds; the ring's gather
crossover against the balanced policy is near the 2 MiB shard limit; on a
pair the one-shot wins through 4 MiB, so `tp2-direct`'s 2 MiB cut is the
capacity's limit, not the fabric's.

`sparknet policy show --profile tp4-ring` prints the policy and its
environment; `CollectivePolicy.from_environment` reads the limits in bytes
or in vLLM's `2MB` syntax; the probe asserts that observed proxy counters
agree with the policy on every timed case.

Changing a backend, channel layout or transport can change floating-point
summation order. The policy preserves fixed-order reduction inside
one-shot; it does not promise batch-invariant output across backends.
