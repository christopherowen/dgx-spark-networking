# dgx-spark-networking

[![validate](https://github.com/christopherowen/dgx-spark-networking/actions/workflows/validate.yml/badge.svg)](https://github.com/christopherowen/dgx-spark-networking/actions/workflows/validate.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Switchless RoCE collectives, measured NCCL profiles and fabric tooling for
inference recipes on DGX Spark clusters. Python import name: `sparknet`.

Two to four Sparks cabled directly through their ConnectX-7 200 GbE ports,
one GPU and one tensor-parallel rank per node, no switch (a switched fabric
of up to sixteen is a configuration path). The library carries the parts of
that fabric that were measured to work best and packages them so a serving
recipe (vLLM, SGLang or a custom engine) can select them by name: a one-shot
RDMA all-reduce and all-gather for the small collectives on the next-token
path, a patched NCCL with a balanced ring policy for the bulk collectives, an
explicit policy for which backend carries which collective, a node map with
validation and discovery, and a probe that qualifies the fabric before a
model is loaded.

## Measured

Four DGX Sparks in a cable loop (GB10, driver 580.178.04, kernel
7.0.0-1019-nvidia-64k, ConnectX-7 firmware 28.45.4028), BF16 all-reduce,
median of the slowest rank per sample of 256 graph-replayed calls:

| Input per rank | NCCL ring, one channel | One-shot, bidirectional host relay |
| --- | ---: | ---: |
| 10 KiB | 90.6 to 92.4 us | 16.9 us |
| 60 KiB | 111.5 to 114.2 us | 28.0 us |
| 480 KiB | 168.9 to 174.1 us | 94.0 us |
| 2 MiB | 359.5 to 362.1 us | 307.7 us |

Two DGX Sparks on one cable (measured 2026-10-05 through this package, same
method, median of three runs; NCCL is the NCCL-only control on the same cable):

| Input per rank | NCCL only | One-shot |
| --- | ---: | ---: |
| 10 KiB | 67.2 to 77.3 us | 10.9 us |
| 60 KiB | 80.0 to 87.8 us | 17.0 us |
| 480 KiB | 109.1 to 122.6 us | 43.1 us |
| 2 MiB | 191.7 to 205.2 us | 137.1 us |

The one-shot collective is one kernel launch, replays inside CUDA graphs and
reduces in fixed rank order, so every rank produces bit-identical output. On
the same ring, the balanced NCCL channel policy took the 2 MiB all-reduce
from 320 to 232 us against the clockwise-only control and put 24.8 to 25.2
percent of RDMA bytes on each of the four interfaces. On three Sparks in a
triangle the direct mode costs 16 to 19 us at 10 KiB. Every number names the
experiment that produced it: [docs/oneshot.md](docs/oneshot.md),
[docs/nccl.md](docs/nccl.md), [docs/provenance.md](docs/provenance.md).

## What is inside

- **One-shot collectives** (`sparknet.oneshot`, RoCEnante in b12x): the one-shot RDMA all-reduce and
  all-gather for the small collectives on the next-token path. One kernel
  launch stages the input into pinned host memory, a proxy posts RDMA writes
  to every peer, the kernel waits on per-lane sequence flags and reduces in
  fixed rank order. Direct (every pair cabled), `ring4` (four nodes in a
  loop, opposite ranks relayed by the neighbours' hosts in both directions)
  and `mesh4` (NIC-forwarded) modes. No GPUDirect RDMA is needed: the GB10's
  unified memory lets the NIC write pinned host memory that the GPU reads in
  place.
- **NCCL profiles** (`sparknet.nccl`, `sparknet/nccl/patches`): the
  environment that keeps NCCL on neighbour edges for the bulk collectives,
  plus the four patches the balanced four-node profile needs (an AArch64
  send-path fence that fixes a hang on every profile, and the bidirectional,
  balanced channel policy). The series ships in the wheel.
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
- **vLLM adapter** (`sparknet.integration.vllm`): a drop-in device
  communicator for the Local Inference Lab vLLM fork, and a
  [reference Dockerfile](docker/Dockerfile) for the fabric parts of a
  serving image.

Recipe builders start with [docs/integration.md](docs/integration.md): fabric
description, image build, probe, engine wiring, checklist and troubleshooting.

## Supported fabrics

| Fabric | Nodes | Transports | Small collectives | Status |
| --- | --- | --- | --- | --- |
| 2x direct connect | 2 | `oneshot-direct`, `nccl-direct` | One-shot over one cable's two PCIe paths; a second cable goes to NCCL (`nccl_hcas`) | measured 2026-10-05 with the triangle's settings (`tp2-direct`): 10 KiB all-reduce 10.9 us, NCCL only 67 to 77 us |
| 3x switchless triangle | 3 | `oneshot-direct`, `nccl-direct` | One-shot direct, every pair cabled | promoted spark-ds41f TP3 recipe (`tp3-triangle`), serving |
| 4x switchless ring | 4 | `oneshot-ring4`, `nccl-ring`, `oneshot-mesh4` | One-shot bidirectional host relay; NCCL on neighbour edges only | promoted spark-ds41f TP4 recipe (`tp4-ring`), serving; NCCL-only control (`tp4-ring-nccl-only`); mesh4 carried, not recommended |
| Switched | 2 to 16 | `oneshot-switched`, `nccl-switched` | One-shot clique over up to two rails; NCCL over every rail with its own topology selection | configuration path (`switched`, `switched-nccl-only`); the clique mode was measured upstream on four switched Sparks, not on this fleet |

The node map says which fabric a site has (`roce_peer_hcas` per peer for
cabled fabrics, `roce_hcas` and `nccl_hcas` rails for a switch), the
transport says which backend carries the small collectives, and the profile
supplies the measured settings. `sparknet topology validate` refuses a map
that does not fit the transport, and `render` refuses a profile that does not
fit the map.

## Status

**Serving.** The deployment repository
([spark-ds41f](https://github.com/christopherowen/spark-ds41f)) promoted its
r6 image on 2026-10-05: it installs this package's 0.2.0 wheel, and its vLLM
fork constructs `sparknet.integration.vllm.SparknetOneShotAllReduce` (its
vLLM patch 0038, the one-file switch described in
[sparknet/integration/vllm/README.md](sparknet/integration/vllm/README.md))
for the TP3 triangle recipe (`oneshot-direct`, `tp3-triangle`) on three
Sparks and the TP4 1M recipe (`oneshot-ring4`, `tp4-ring`) on the four-node
ring. Both were benchmarked at their recipe limits on that image before
promotion.

**Vendored, verified.** The one-shot sources are vendored verbatim from the
hardware-qualified tree of spark-ds41f (b12x `f8069b2c` plus its eleven
switchless patches, tree `cd615bd6`), with only imports, environment names
and preparation decoupled from b12x; the protocol, kernels, proxy and wire
ABI (10) are unchanged, and the C proxy runs under sanitizers in CI through
26,000 simulated collectives.

**Qualified through this package.** On 2026-10-05 the package's own GPU
suite and collective probe ran on the fleet for the first time, on the
two-Spark pair and the four-node ring, in the r6 image: both passed, the
pair was measured for the first time (`tp2-direct` keeps the triangle's
settings and is now measured, not tuned), and the receipts are under
`evidence/2026-10-05-tilelang-port`.

**Two kernel families.** The one-shot kernels now exist twice: the vendored
CuTe DSL kernels and a TileLang port (`SPARKNET_ROCE_KERNELS=tilelang`) that
generates CUDA source with the protocol's device side in one header. The
GPU suite shows the two bit-identical on the pair and the ring; latencies
are equal within noise, with TileLang 3 to 5 percent faster at 480 KiB.
Serving was benchmarked on both (2026-10-05, [docs/oneshot.md](docs/oneshot.md)):
TileLang is 0.6 to 1.2 percent slower per decode step in the lean screen,
equal elsewhere, so by the promotion rule (measurably better in serving) the
default stays `cute` while the cause is found.

**Not yet.** The switched profiles reuse the triangle's settings and say so
in their status. The GPU-initiated transport is staged, not implemented;
the TileLang port is the kernel it will be written into.

## Install

```sh
pip install git+https://github.com/christopherowen/dgx-spark-networking@v0.3.0
```

The package has no dependencies: the CLI and the topology, profile, policy
and probe-planning modules run on any machine. On the nodes, the one-shot
runtime and the collective probe need torch, cuda-python and the CuTe DSL
pinned by the `runtime` extra (`pip install 'dgx-spark-networking[runtime] @ git+...'`);
inside a vLLM image that already ships them, install with `--no-deps`.

## Quick start

On any machine:

```sh
sparknet topology example tp4-ring > nodes.json        # or tp2-direct, tp3-triangle, switched; edit for your site
sparknet topology validate nodes.json --transport oneshot-ring4
sparknet topology render nodes.json dgx3 --transport oneshot-ring4 --profile tp4-ring
sparknet nccl profiles                                 # every profile, its status and the patches it needs
sparknet nccl validate --profile tp4-ring
sparknet nccl patches --export ./nccl-patches          # the series, for an image build
sparknet policy show --profile tp4-ring
```

On a Spark, with serving stopped:

```sh
sparknet topology discover dgx1 dgx2 dgx3 dgx4 --out site --management-ip dgx1=192.0.2.1 ...   # LLDP, read-only
sparknet probe doctor nodes.json --node dgx3 --transport oneshot-ring4                          # live links, GID, MTU, memlock
sparknet probe render-command nodes.json dgx3 --transport oneshot-ring4 \
  --profile tp4-ring --image <serving image> -- --benchmark --counter-samples                  # one bounded container per rank
sparknet probe gpudirect
```

`discover` reads the cabling over LLDP and writes `nodes.json` and each
node's `40-cx7.yaml` under the fleet's `10.<a><b>.<path>.<N>` addressing;
`--fabric switched` reads each host's active rails from sysfs instead.

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

[examples/minimal_engine.py](examples/minimal_engine.py) is the complete
pattern in ninety lines.

## Questions

**I have two Sparks and one cable.** Use `tp2-direct` with `oneshot-direct`.
Measured on this fleet's pair on 2026-10-05: a 10 KiB BF16 all-reduce in
10.9 us against 67 to 77 us for NCCL on the same cable, a 2 MiB one in 137 us
against 192 to 205 us ([docs/oneshot.md](docs/oneshot.md)). The settings are
the triangle's, not tuned for a pair. Run the probe with `--benchmark` before
serving, and consider [reporting the result](.github/ISSUE_TEMPLATE/fabric_report.yml).
A second cable on the other port goes to NCCL through `nccl_hcas`.

**Do I need a switch, GPUDirect RDMA or `nvidia-peermem`?** No. Cabled
fabrics need no switch, and the protocol registers pinned host memory with a
plain `ibv_reg_mr` that the GB10 reads in place. A switch is supported as a
configuration path and usually needs lossless RoCE (`roce_traffic_class`).

**Which NCCL?** 2.30.7, rebuilt for SM121 with the packaged series. Patch
0001 (the AArch64 send-path fence, NVIDIA/nccl#2393) is required on every
profile; without it NCCL can hang every rank. 0002 to 0004 are the ring
profile's balanced channel policy. `scripts/build-nccl.sh` and the reference
Dockerfile build it; `sparknet nccl validate --unpatched` tells you which
profile settings an unpatched library would ignore.

**Does it work with upstream vLLM, or SGLang?** The adapter replaces a class
in the Local Inference Lab vLLM fork. The integration point is the same in
upstream vLLM and in SGLang (the custom all-reduce of the CUDA device
communicator), and the runtime API is engine-independent, but no patch ships
for either; [docs/integration.md](docs/integration.md) section 5 is the
contract an engine must keep.

**Is the output deterministic?** The one-shot all-reduce reduces in fixed
rank order, so all ranks produce bit-identical output and it is identical
across runs. Reversing or re-partitioning NCCL channels can change
floating-point summation order for the bulk collectives; no bitwise
equivalence with the one-direction rings is claimed, and the policy does not
promise batch-invariant output across the two backends.

**What does a node need?** DGX Spark OS 26.09 or later with `kho=off`,
driver 580.178.04, rdma-core 50 with `libibverbs-dev` and a C compiler for
the proxy, MTU 9000 with one static IPv4 address per cable stripe, GID index
3 as the IPv4 RoCE v2 GID, unlimited locked memory:
[docs/host-prerequisites.md](docs/host-prerequisites.md). `sparknet probe
doctor` checks all of it and changes nothing.

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

## Layout

```text
sparknet/              the library (CPU-only subpackages never import torch)
sparknet/oneshot/      the runtime, the proxy, and both kernel families (CuTe DSL and TileLang)
sparknet/nccl/patches/ NCCL 2.30.7 patch series and README, shipped in the wheel
native/gpunetio/       DOCA GPUNetIO pin, Spark patches and build script
recipes/               rendered environments for the named profiles
docker/                reference Dockerfile for the fabric parts of a serving image
examples/              a minimal engine that uses the runtime the intended way
docs/                  design, topology, nccl, oneshot, policy, roadmap, provenance
tests/                 unit tests, the C proxy simulator, tests/gpu (torchrun)
benchmarks/            one-shot versus NCCL latency with a receipt
evidence/              probe receipts and GPU-test summaries behind the numbers
scripts/               NCCL build and the fleet GPU-test launcher
upstreams.lock.json    pinned revisions, patch heads and tree hashes
```

Tests: `make test` (stdlib unittest; the simulator needs a C compiler) and
`make lint` (ruff). CI runs both on Python 3.10, 3.12 and 3.14, installs the
wheel and drives the CLI from it, re-applies the NCCL series to the pinned
release and checks the tree hash, and lints the Dockerfile. Everything in
`docs/` states what was measured, where, and what remains unqualified.

## Contributing

Results from other fabrics are welcome, especially from pairs and from anyone
with a switch: the
[fabric report](.github/ISSUE_TEMPLATE/fabric_report.yml) template lists
what to include. [CONTRIBUTING.md](CONTRIBUTING.md) has the change
discipline; [CHANGELOG.md](CHANGELOG.md) the release history.

## License

Apache-2.0. Third-party attribution is in `NOTICE`. The project is not
affiliated with NVIDIA.
