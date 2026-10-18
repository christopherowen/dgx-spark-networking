# Design

## The fabric

Each DGX Spark exposes its single cabled QSFP port as two PCIe Gen5 x4
functions (`rocep1s0f0` and `roceP2p1s0f0` on port 0, `rocep1s0f1` and
`roceP2p1s0f1` on port 1), 200 Gb/s per port, RoCE v2 over IPv4. Each
function's source GID is its port's one IPv4 RoCE v2 GID in the cable's subnet,
found by address when the runtime opens the device (see "GIDs by address"
below); no GID slot is configured. Port 0 cables to the next node, port 1 to the previous one. Two nodes
share one cable (a second cable on the other port is extra NCCL bandwidth);
three nodes form a triangle (every pair cabled); four form a loop, where the
opposite rank has no cable; behind a switch every rank reaches every other
over the same rails, up to sixteen in the proxy's geometry. The GB10 is an integrated GPU with unified
memory: the NIC registers pinned host memory with a plain `ibv_reg_mr` and
the GPU reads it in place at full bandwidth, so no GPUDirect RDMA is needed
for the protocol to run.

## The two backends and the policy

Decode is latency-bound: the per-step all-reduces are tens of KiB and the
MTP logits gathers a few hundred KiB. NCCL's ring costs 75 to 90 us at 10 KiB
on this fabric; the one-shot kernel costs 16 to 19 us. Prefill and
the sequence-parallel reduce-scatters are bandwidth-bound and NCCL's ring
moves 1.5 payloads per rank where the one-shot moves three; NCCL wins above
about 1 MiB. So the policy is two backends with a fixed, measured crossover
and a fail-stop contract (`docs/policy.md`).

## One-shot protocol

One pinned region per rank: `recv[src][slot]`, `flag[src][slot][lane]`,
`send[slot]` and a control record. One kernel launch per collective:

1. stage the input into `send[seq & 1]`;
2. the last block to finish staging publishes `nbytes` (per slot) and `seq`
   to the control record that the proxy polls; the proxy divides each peer
   payload into one stripe per route lane and posts each stripe followed by
   a 4-byte `seq` write on the same reliable QP, so a flag cannot land before
   its data;
3. the kernel waits on `flag[peer][seq & 1][lane] == seq` for every peer and
   lane, bounded by the timeout in the control record; a timeout records the
   missing peer, its lane and the time waited, then stops the GPU;
4. all-reduce sums the local input and every peer slot in fixed rank order
   (bit-identical across ranks); all-gather writes the concatenated layout
   directly;
5. the last block to finish advances a device-resident epoch, so the sequence
   number is a runtime value and graph replay is correct.

Two slots suffice because a peer cannot start op k+2 before finishing k+1,
which needs our k+1 data, posted only after our op k kernel completed.

### ring4

Four nodes in a loop: each rank writes its payload to both neighbours over
their direct QPs. For the opposite rank, each physical stripe is split into
disjoint 16-byte packs; one half travels clockwise and the other
counterclockwise, each forwarded by the intermediate host as soon as its own
first-hop flag arrives (an AArch64 `dmb oshld` after observing the NIC's
flag, before the forwarding DMA). The destination indexes the relayed slot
as the original source, so buffer layout and reduction order are unchanged.
Every fragment, including an empty one, has its own flag; the opposite GPU
waits on twice as many lanes. Measured: equal clockwise and counterclockwise
bytes, 25 percent per interface, near-baseline latency.

### mesh4

Opposite ranks reached through ConnectX-7 hardware forwarding on the
intermediate node (flow label 16383 marks the opposite-peer QPs; a host
marker rewrites the EtherType and a TC rule forwards it). The proxy supports
it; the host-side marker, routes and TC rules are not part of this library.
It is retained for comparison, not recommended: large payloads showed
receive-buffer overflows and retransmissions and poorer latency than the
host relay, and its fabric lifecycle needs a supervisor.

## Transport boundary

`sparknet.transport.Transport` is the posting side of the protocol: open the
HCAs, register the region, exchange queue-pair records, start posting on
doorbells, report fail-stop state. The host proxy implements it today. The
GPU side (slots, flags, epoch, reduction) is independent of who posts; the
GPUNetIO path keeps the geometry and the wire layout and moves WQE
construction into the kernel. See `docs/gpudirect-roadmap.md`.

## Topology and rendering

`nodes.json` describes cables only: per node, which local HCAs reach which
peer rank in cable order and the `/24` of each cable stripe. Validation
checks rank order, reciprocity, distinct interfaces, one stripe count,
subnets at exactly two endpoints and (for ring transports) neighbour-only
routes. Rendering produces the per-node environment: the JSON peer map,
`NCCL_IB_HCA` with exact names, the one-shot HCAs' cable subnets and the
ring policy for NCCL. The runtime refuses geometry that differs between ranks
at connect time, so a wrong map fails before serving starts.

### GIDs by address

A ConnectX-7 port's GID table holds the link-local default and, for its
address, a RoCE v1 and a RoCE v2 entry; with the fleet's netplan that is
index 3 for the IPv4 RoCE v2 GID. The kernel files a re-added address in the
lowest free slot, and a slot whose entry RDMA resources still reference is not
free. A link flap while ranks hold queue pairs (a neighbour rebooting under a
running service) therefore re-adds the address at index 4 and leaves 3 empty,
and a fixed index then names an empty slot until the interface is re-activated
with nothing using RDMA.

The proxy takes each HCA's source GID from what the entry is: it scans the
port's table with `ibv_query_gid_ex` and takes the one entry that is RoCE v2,
IPv4-mapped and inside that HCA's `SPARKNET_ROCE_SUBNETS` network (rendered
from `roce_subnets`; without it, any IPv4 address). None, or more than one, is a
setup error naming the device and the subnet, never a fallback to a slot. This
is NCCL's own selection with `NCCL_IB_GID_INDEX` unset
(`ncclIbGetGidIndex` scans by `NCCL_IB_ADDR_FAMILY`, `NCCL_IB_ADDR_RANGE` and
`NCCL_IB_ROCE_VERSION_NUM`), made strict, and what librdmacm does when it binds
a source address. The profiles set `NCCL_IB_ADDR_FAMILY=AF_INET` and
`NCCL_IB_ROCE_VERSION_NUM=2` and no index, so both libraries in a process
select the same GID; `sparknet probe doctor` checks that the port carries
exactly one IPv4 RoCE v2 GID, in the cable's subnet, which makes NCCL's
lowest-index choice and the proxy's choice the same entry. A running
process keeps the slot it connected with; after a flap its queue pairs fail
and the restart finds the moved entry.
