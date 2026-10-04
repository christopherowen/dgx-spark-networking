# Topology

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
  RoCEnante and NCCL; it may differ between nodes.
- `management_interface` renders `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`
  and `TP_SOCKET_IFNAME`.

Transports: `rocenante-direct` (two or three nodes, every pair cabled),
`nccl-ring`, `rocenante-ring4` and `rocenante-mesh4` (four nodes, neighbours
only). `sparknet topology validate` reports every configuration mistake;
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
`ip -br a`, or discovery.

## Rendering

`sparknet topology render nodes.json dgx3 --transport rocenante-ring4 --profile tp4-ring`
prints the rank's complete environment: the profile's NCCL and RoCEnante
settings, the ring policy (`NCCL_ALGO=Ring`, runtime connect with cuMem,
trees, PAT, NVLS, CollNet and GIN off, subnet-aware routing over merged
NICs), `NCCL_IB_HCA` with exact device names, the JSON peer map and the GID
index. With `--no-compat` only the `SPARKNET_*` names are emitted; by
default the `B12X_ROCE_*` and `VLLM_*` aliases are included so the same
environment drives the current spark3 image.
