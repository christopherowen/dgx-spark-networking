# Topology

## The four fabrics

| Fabric | Map fields | Transports |
| --- | --- | --- |
| Two nodes on one cable (optionally a second cable on the other port) | `roce_peer_hcas` with one peer; `nccl_hcas` lists the second cable's functions; subnets `10.12.1/2` and, for the second cable, `10.12.3/4` | `oneshot-direct`, `nccl-direct` |
| Three nodes in a triangle | `roce_peer_hcas` with both peers, port 0 to the next node, port 1 to the previous | `oneshot-direct`, `nccl-direct` |
| Four nodes in a loop | `roce_peer_hcas` with the two neighbours only, ranks in cable order | `oneshot-ring4`, `nccl-ring`, `oneshot-mesh4` |
| Two to sixteen nodes behind a switch | `roce_hcas` (one or two rails for one-shot), `nccl_hcas` (every rail), one subnet per rail shared by all nodes, optional `roce_traffic_class` | `oneshot-switched`, `nccl-switched` |

One-shot stripes one peer payload over at most two local functions, so on a
two-cable pair or a dual-rail switch it uses one port's two PCIe paths and
NCCL uses everything in `nccl_hcas`. On a switched fabric the runtime runs
its clique mode: the same rails reach every peer, and NCCL keeps its own
topology and algorithm selection (trees allowed). A switch usually needs
lossless RoCE (PFC or ECN); `roce_traffic_class` renders `NCCL_IB_TC` and
`SPARKNET_ROCE_TRAFFIC_CLASS` so both backends mark packets the same way.
The examples live under `sparknet/topology/examples/`.

## Node map

```json
{
  "schema_version": 1,
  "ssh_user": "spark",
  "nodes": [
    {"name": "dgx1", "rank": 0, "management_ip": "192.0.2.1", "head": true,
     "roce_gid_index": 3, "management_interface": "enP7s7",
     "roce_peer_hcas": {"3": ["rocep1s0f1", "roceP2p1s0f1"], "1": ["rocep1s0f0", "roceP2p1s0f0"]},
     "roce_subnets": {"rocep1s0f0": "10.12.1.0/24", "roceP2p1s0f0": "10.12.2.0/24",
                      "rocep1s0f1": "10.14.1.0/24", "roceP2p1s0f1": "10.14.2.0/24"}}
  ]
}
```

- `rank` is the position in the cable loop; the API head is rank 0.
- `roce_peer_hcas` lists the local HCAs reaching each peer, one or two
  stripes per link with matching counts at both endpoints and the same
  stripe position on both ends. A local HCA belongs to one peer only.
- `roce_subnets` records the IPv4 `/24` of each cable stripe; each subnet
  appears at exactly its two endpoints. The two PCIe functions on one cable
  use two subnets.
- `roce_gid_index` selects the node's IPv4 RoCE v2 GID slot for both
  one-shot and NCCL; it may differ between nodes.
- `management_interface` renders `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`
  and `TP_SOCKET_IFNAME`.

`sparknet topology validate` reports every configuration mistake;
`sparknet probe doctor` adds the live checks on a node (device present and
active, GID in the declared subnet, MTU 9000, `kho=off`, memlock, compiler
and headers). Neither proves that a cable delivers RDMA packets; the
collective probe does.

## Addressing and cabling

The fleet rule: cable between nodes `a < b`, path 1 (`enp1s0*`) on
`10.<a><b>.1.0/24`, path 2 (`enP2p1s0*`) on `10.<a><b>.2.0/24`, host address
is the node number. MTU 9000, static, no DHCP, no IPv6 (so GID index 3 stays
the IPv4 address). `sparknet topology discover` collects LLDP neighbour MACs
over SSH, resolves the cables, and writes `40-cx7.yaml` per node plus
`nodes.json` in cable order from the head. It changes nothing on a host.

The cabling flips between a triangle (TP3) and a loop (TP4) by moving one
cable. Always check live links before trusting a map: `rdma link`,
`ip -br a`, or discovery. Behind a switch LLDP sees the switch, not the
peers, so `discover --fabric switched` reads each host's rails from sysfs
and takes the rail subnets from the live addresses.

## Rendering

`sparknet topology render nodes.json dgx3 --transport oneshot-ring4 --profile tp4-ring`
prints the rank's complete environment: the profile's NCCL and one-shot
settings, the ring policy (`NCCL_ALGO=Ring`, runtime connect with cuMem,
trees, PAT, NVLS, CollNet and GIN off, subnet-aware routing over merged
NICs), `NCCL_IB_HCA` with exact device names, the JSON peer map and the GID
index, each under its one `SPARKNET_ROCE_*` or `NCCL_*` name.
