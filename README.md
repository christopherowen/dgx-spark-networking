# dgx-spark-networking

Switchless RoCE collectives, measured NCCL profiles and fabric tooling for
inference recipes on DGX Spark clusters. Python import name: `sparknet`.

Two to four Sparks cabled directly through their ConnectX-7 200 GbE ports,
one GPU and one tensor-parallel rank per node, no switch. The library carries
the parts of that fabric that were measured to work best and packages them so
a serving recipe (vLLM, SGLang or a custom engine) can select them by name:

- **One-shot collectives** (`sparknet.oneshot`, RoCEnante in b12x): the one-shot RDMA all-reduce and
  all-gather for the small collectives on the next-token path. One kernel
  launch stages the input into pinned host memory, a proxy posts RDMA writes
  to every peer, the kernel waits on per-lane sequence flags and reduces in
  fixed rank order, so all ranks produce bit-identical output and the
  collective replays inside CUDA graphs. Direct (every pair cabled), `ring4`
  (four nodes in a loop, opposite ranks relayed by the neighbours' hosts in
  both directions) and `mesh4` (NIC-forwarded) modes.
- **NCCL profiles** (`sparknet.nccl`, `patches/nccl`): the environment that
  keeps NCCL on neighbour edges for the bulk collectives, plus the four
  patches the balanced four-node profile needs (an AArch64 send-path fence
  and the bidirectional, balanced channel policy).
- **Policy** (`sparknet.policy`): the explicit, rank-invariant decision of
  which backend carries which collective, with dispatch, capacity and
  all-gather limits kept distinct and fail-stop semantics.
- **Topology** (`sparknet.topology`): the node map (`nodes.json`) that
  describes cables, validation before any queue pair opens, per-node
  environment rendering, and LLDP discovery that generates the map and the
  netplan files from the cabling.
- **Probe** (`sparknet.probe`): a read-only fabric doctor, the model-free
  collective correctness and latency probe with RDMA and port counters, and
  a report of what the host offers for the GPU-initiated transport.
- **Transport boundary** (`sparknet.transport`): the seam where the host
  proxy is replaced by DOCA GPUNetIO. The CPU-proxy GPUNetIO path is
  qualified on the fleet's links and its patches live in `native/gpunetio`;
  see [docs/gpudirect-roadmap.md](docs/gpudirect-roadmap.md).

Recipe builders start with [docs/integration.md](docs/integration.md): fabric
description, image build, probe, engine wiring, checklist and troubleshooting.

## Supported fabrics

| Fabric | Nodes | Transports | Small collectives | Status |
| --- | --- | --- | --- | --- |
| 2x direct connect | 2 | `oneshot-direct`, `nccl-direct` | One-shot over one cable's two PCIe paths; a second cable goes to NCCL (`nccl_hcas`) | configuration path with the triangle's settings (`tp2-direct`); not measured on this fleet |
| 3x switchless triangle | 3 | `oneshot-direct`, `nccl-direct` | One-shot direct, every pair cabled | promoted spark3 TP3 baseline (`tp3-triangle`) |
| 4x switchless ring | 4 | `oneshot-ring4`, `nccl-ring`, `oneshot-mesh4` | One-shot bidirectional host relay; NCCL on neighbour edges only | measured balanced candidate (`tp4-ring`), NCCL-only control (`tp4-ring-nccl-only`); mesh4 carried, not recommended |
| Switched | 2 to 16 | `oneshot-switched`, `nccl-switched` | One-shot clique over up to two rails; NCCL over every rail with its own topology selection | configuration path (`switched`, `switched-nccl-only`); the clique mode was measured upstream on four switched Sparks, not on this fleet |

The node map says which fabric a site has (`roce_peer_hcas` per peer for
cabled fabrics, `roce_hcas` and `nccl_hcas` rails for a switch), the
transport says which backend carries the small collectives, and the profile
supplies the measured settings. `sparknet topology validate` refuses a map
that does not fit the transport, and `render` refuses a profile that does not
fit the map.

## Status

The one-shot sources are vendored verbatim from the hardware-qualified tree
of spark3-vllm-ds41f (b12x `f8069b2c` plus its eleven switchless patches,
tree `cd615bd6`, the `roce-balanced-dispatch-v1` serving image), with only
imports, environment names and preparation decoupled from b12x; the protocol,
kernels, proxy and wire ABI (10) are unchanged. The TP3 profile is the
promoted spark3 baseline; the TP4 profile is the measured balanced candidate
(every interface at 24.8 to 25.2 percent of RDMA bytes, 2 MiB all-reduce 320
to 232 us, prefill +1.6 to +1.9 percent, decode within noise). Numbers and
their experiments are in [docs/nccl.md](docs/nccl.md) and
[docs/oneshot.md](docs/oneshot.md).

What has not happened yet: the vendored runtime has not been run on the
Sparks through this package (the GPU test and the probe exist for that), the
vLLM fork still imports `b12x.comm.roce` (the one-file switch is described in
[sparknet/integration/vllm/README.md](sparknet/integration/vllm/README.md)),
and the GPU-initiated transport is staged, not implemented.

## Direction: GPU-initiated networking

The host proxy is the measured path today, and the library is built so that
the posting side can move into the GPU without touching the protocol:
`sparknet.transport` is the seam, `Geometry` is what a transport needs, and
the kernels, flags, two-slot lifetime, fixed-rank reduction and fail-stop
contract stay as they are.

1. **GPUNetIO with the CPU-proxy handler** (next): the collective kernel
   builds the RDMA write and flag WQEs itself right after staging and
   publishes them with a system-scope release; a host thread only rings the
   doorbell. DOCA GPUNetIO open source builds for SM121 on the stock Spark
   stack and, with the two patches in `native/gpunetio` (shared host memory,
   system-scope fence), its write-latency sample ran on every link of four
   Sparks at 4.1 to 7 us half round trip. `sparknet probe gpudirect` reports
   whether a host is ready for it.
2. **GPU doorbell**: the kernel rings the NIC's UAR directly. The one attempt
   on the fleet ended in a host failure (one node rebooted, another's GPU
   needed a reboot) with nothing classified in the logs, so this stage waits
   for a host-level investigation on an idle node.
3. **Device-memory registration**: rdma-core 50 on the fleet exports
   `ibv_reg_dmabuf_mr`; whether the GB10 driver exports a dma-buf for device
   allocations, and whether `nvidia-peermem` (present, not loaded) is the
   better route, is what the probe checks. On unified memory this is about
   placement, not necessity.

The full plan with acceptance criteria is
[docs/gpudirect-roadmap.md](docs/gpudirect-roadmap.md).

## Quick start

```sh
sparknet topology example tp4-ring > nodes.json        # or tp2-direct, tp3-triangle, switched
sparknet topology validate nodes.json --transport oneshot-ring4
sparknet topology render nodes.json dgx3 --transport oneshot-ring4 --profile tp4-ring
sparknet nccl validate --profile tp4-ring
sparknet probe doctor nodes.json --node dgx3 --transport oneshot-ring4      # on the node
sparknet probe render-command nodes.json dgx3 --transport oneshot-ring4 \
  --profile tp4-ring --image <serving image> -- --benchmark --counter-samples
sparknet probe gpudirect
```

`sparknet topology discover dgx1 dgx2 dgx3 dgx4 --out site --management-ip dgx1=192.0.2.1 ...`
reads the cabling over LLDP (read-only) and writes `nodes.json` and each
node's `40-cx7.yaml` under the fleet's `10.<a><b>.<path>.<N>` addressing;
`--fabric switched` reads each host's active rails from sysfs instead and
derives the rail subnets from their live addresses.

In an engine:

```python
from sparknet import oneshot
from sparknet.policy import TP4_POLICY

runtime = oneshot.AllReduce.from_exchange_group(
    exchange_group=cpu_group, device=device,
    max_size=TP4_POLICY.all_reduce_capacity_bytes,
    max_gather_bytes=TP4_POLICY.all_gather_shard_bytes)
runtime.prepare((torch.bfloat16, torch.float32), padded_gather=True)
...
if runtime.should_allreduce(x):          # rank-invariant, dispatch limit
    y = runtime.all_reduce(x)
runtime.check_health()                   # after the step's own host sync
```

## Layout

```text
sparknet/            the library (CPU-only subpackages never import torch)
patches/nccl/        NCCL 2.30.7 patch series and README
native/gpunetio/     DOCA GPUNetIO pin, Spark patches and build script
recipes/             rendered environments for the named profiles
examples/            a minimal engine that uses the runtime the intended way
docs/                design, topology, nccl, oneshot, policy, roadmap, provenance
tests/               unit tests, the C proxy simulator, tests/gpu (torchrun)
benchmarks/          one-shot versus NCCL latency with a receipt
upstreams.lock.json  pinned revisions, patch heads and tree hashes
```

Tests: `make test` (stdlib unittest; the simulator needs a C compiler).
Everything in `docs/` states what was measured, where, and what remains
unqualified.

## License

Apache-2.0. Third-party attribution is in `NOTICE`.
