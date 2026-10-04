# NCCL patch stack

Base: `NVIDIA/nccl@73cf112295c33aee2b895f329f592f2a9b4b0f97` (release tag
`v2.30.7-1`), the version the vLLM nightly base image ships as the
`nvidia-nccl-cu13` wheel. `scripts/build-nccl.sh` rebuilds it for SM121 with
this series and writes `libnccl.so.2` plus its SHA-256; a serving image
replaces the wheel's library with it and checks the hash at import.

- `0001-ib-cts-nreqs-acquire-fence.patch` (Stanislav Bardyuk, NVIDIA/nccl#2393)
  orders the clear-to-send `nreqs` load after the `idx` check in `ncclIbIsend`.
  On AArch64 the two loads can be reordered, so the proxy thread can spin
  forever on a request the receiver never writes: a hang with every rank
  waiting. One acquire fence on the IB send path. Required on every profile.
- `0002-bidirectional-switchless-rings.patch` adds `NCCL_SWITCHLESS_BIDIRECTIONAL`:
  after channel duplication, reverse half of the final ring channels so a
  neighbour ring carries bulk traffic in both directions. Fails closed unless
  the fabric is three or four nodes, one rank per node, an even channel count
  and rings already in cable/rank order.
- `0003-balanced-channel-allocation.patch` adds mode 2 (reverse channels 1 and
  2 of each four-channel group, so the first pair already spans both
  directions and a full group uses both NIC roots both ways) and exposes the
  32 KiB allocation-cell floor as `NCCL_MIN_TRAFFIC_PER_CHANNEL`.
- `0004-adaptive-small-ring-threads.patch`: in mode 2, a tiny Ring call tries
  fewer threads down to 128 before dropping channels, so four direction/root
  lanes stay active at 128 bytes per rank without the measured cost of
  forcing 128 threads globally.

Measured together as the `tp4-ring` profile (spark3-vllm-ds41f
`experiments/2026-10-03-balanced-policy`): 24.8 to 25.2 percent of RDMA
bytes on each of the four interfaces, clockwise share 49.7 to 50.2 percent,
2 MiB BF16 all-reduce 320 to 232 us against the clockwise control. Reversing
or re-partitioning channels can change floating-point summation order; no
bitwise equivalence with the one-direction rings is claimed.

Upstream status: 0001 is open upstream; 0002 to 0004 are local, not submitted.
