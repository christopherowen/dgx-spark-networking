# NCCL

NCCL carries the bulk collectives: all-reduces above the dispatch limit,
all-gathers above the shard limit, every reduce-scatter and the variable
collectives. Pinned source: NCCL 2.30.7 (`v2.30.7-1`), rebuilt for SM121
with `patches/nccl` by `scripts/build-nccl.sh`.

## Profiles

| Setting | `tp2-direct`, `tp3-triangle`, `switched` | `direct-nccl-only`, `switched-nccl-only` | `tp4-ring` | `tp4-ring-nccl-only` |
| --- | --- | --- | --- | --- |
| Physical fabric | one cable, triangle, or a switch | same | four-node loop | four-node loop |
| Small collectives | One-shot direct (clique) | NCCL | One-shot ring4 relay | NCCL |
| NCCL algorithm | upstream selection | upstream selection | Ring, neighbours only | Ring, neighbours only |
| Channels (min/max) | upstream / 8 | upstream / 8 | 4 / 4 | 4 / 4 |
| Buffer | 1 MiB | 1 MiB | 4 MiB | 1 MiB |
| LL128 | 256 KiB buffer, protocol excluded | same | same | same |
| Direction policy | upstream | upstream | mode 2: both directions, both roots | upstream (clockwise) |
| Allocation floor | upstream (32 KiB) | upstream | 512 bytes | upstream |
| Thread thresholds | upstream | upstream | `-2 -2 -2 1 1 1` | upstream |
| cuMem / runtime connect | off | off | on (required for neighbour-only connects) | on |
| One-shot all-reduce dispatch / capacity | 2 MiB / 2 MiB | none | 1 MiB / 2 MiB | none |
| One-shot all-gather shard | 4 MiB | none | 2 MiB | none |

Only `tp3-triangle` and `tp4-ring` were measured on this fleet. `tp2-direct`
and the two switched profiles reuse the triangle's settings because the
runtime's direct mode is the same clique protocol; their status fields say
so, and a site adopting them owes itself the collective probe before serving.

`sparknet nccl env --profile <name>` prints the environment,
`sparknet nccl validate` checks it (patch controls on an unpatched library,
contradictory channels, thresholds, limits), `sparknet nccl profiles` lists
the patches each profile needs.

## Why the ring settings

`NCCL_ALGO=Ring` with ranks in cable order keeps traffic on neighbour edges
(`src/graph/connect.cc:connectRings`). `NCCL_RUNTIME_CONNECT=1` postpones
connections until an algorithm is used, and in 2.30.7 requires cuMem
support, so `NCCL_CUMEM_ENABLE=1`; with eager connects the communicator
would try to connect trees and PAT to the unreachable opposite rank. PAT,
NVLS, CollNet, MNNVL, RMA and GIN are disabled. NIC merging with
subnet-aware routing lets NCCL pick the local function that matches the
neighbour's advertised subnet; the renderer supplies exact device names.

The upstream ring runs every channel clockwise: physical-port counters on the
unpatched four-channel profile showed about 99.5 to 0.5 bulk transmission.
Patch 0002 reverses half the channels; patch 0003's mode 2 reverses channels
1 and 2 of each four-channel block so even reduced-channel calls use both
directions and both NIC roots, and exposes the allocation floor; patch 0004
keeps four lanes active for tiny calls by shrinking thread blocks before
dropping channels.

## Measurements (spark3-vllm-ds41f, four Sparks, 2026-10-03)

Balanced selected policy against the clockwise control, same image, BF16,
median of the slowest rank per sample:

| Operation and size | Control | Balanced | Change |
| --- | ---: | ---: | ---: |
| All-reduce, 2 MiB | 320 us | 232 us | -27.4% |
| All-gather, 4 MiB | 706 us | 618 us | -12.5% |
| All-gather, 10 MiB | 1,550 us | 1,426 us | -8.0% |
| Reduce-scatter, 10 MiB | 1,571 us | 1,431 us | -8.9% |
| Reduce-scatter, 128 B | 48 us | 65 us | +34.3% |
| Reduce-scatter, 480 KiB | 122 us | 152 us | +24.1% |

Every interface carried 24.8 to 25.2 percent of RDMA bytes; clockwise share
49.7 to 50.2 percent. Four channels against one channel improved the 4K to
64K filler prefill by about 23 percent in serving; eight channels added 0.7
to 1.2 percent and tripped the thermal guard, so four is selected. A 256 KiB
buffer hurt bulk traffic. Decode TPS differences had intervals containing
zero; the repeated longer-prefill gain was +1.6 to +1.9 percent. These are
transport screens and one cooled serving screen (experiments
`2026-10-03-balanced-policy`, `2026-10-03-collective-serving`,
`2026-10-03-nccl-bidirectional`), not sustained thermal qualification.

Reversing or re-partitioning channels can change floating-point summation
order; no bitwise equivalence with the clockwise rings is claimed.

## The fence (patch 0001)

`ncclIbIsend` loads the clear-to-send slot's `nreqs` after checking `idx`;
AArch64 may reorder the loads and read a previous round's value, so the
proxy thread spins forever and every rank hangs. One acquire fence on the
path where the CTS has arrived. It costs nothing measurable (spark3
`2026-09-27-nccl-fence`) and is required on every profile. Upstream:
NVIDIA/nccl#2393.
