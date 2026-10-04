# dgx-spark-networking

Switchless RoCE collectives, measured NCCL profiles and fabric tooling for
inference recipes on DGX Spark clusters. Python import name: `sparknet`.

Two to four Sparks cabled directly through their ConnectX-7 200 GbE ports,
one GPU and one tensor-parallel rank per node, no switch. The library carries
the parts of that fabric that were measured to work best and packages them so
a serving recipe (vLLM, SGLang or a custom engine) can select them by name:

- **RoCEnante** (`sparknet.rocenante`): the one-shot RDMA all-reduce and
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

## Status

The RoCEnante sources are vendored verbatim from the hardware-qualified tree
of spark3-vllm-ds41f (b12x `f8069b2c` plus its eleven switchless patches,
tree `cd615bd6`, the `roce-balanced-dispatch-v1` serving image), with only
imports, environment names and preparation decoupled from b12x; the protocol,
kernels, proxy and wire ABI (10) are unchanged. The TP3 profile is the
promoted spark3 baseline; the TP4 profile is the measured balanced candidate
(every interface at 24.8 to 25.2 percent of RDMA bytes, 2 MiB all-reduce 320
to 232 us, prefill +1.6 to +1.9 percent, decode within noise). Numbers and
their experiments are in [docs/nccl.md](docs/nccl.md) and
[docs/rocenante.md](docs/rocenante.md).

What has not happened yet: the vendored runtime has not been run on the
Sparks through this package (the GPU test and the probe exist for that), the
vLLM fork still imports `b12x.comm.roce` (the one-file switch is described in
[sparknet/integration/vllm/README.md](sparknet/integration/vllm/README.md)),
and the GPU-initiated transport is staged, not implemented.

## Quick start

```sh
sparknet topology example tp4-ring > nodes.json        # edit for the site
sparknet topology validate nodes.json --transport rocenante-ring4
sparknet topology render nodes.json dgx3 --transport rocenante-ring4 --profile tp4-ring
sparknet nccl validate --profile tp4-ring
sparknet probe doctor nodes.json --node dgx3 --transport rocenante-ring4      # on the node
sparknet probe render-command nodes.json dgx3 --transport rocenante-ring4 \
  --profile tp4-ring --image <serving image> -- --benchmark --counter-samples
sparknet probe gpudirect
```

`sparknet topology discover dgx1 dgx2 dgx3 dgx4 --out site --management-ip dgx1=10.0.1.71 ...`
reads the cabling over LLDP (read-only) and writes `nodes.json` and each
node's `40-cx7.yaml` under the fleet's `10.<a><b>.<path>.<N>` addressing.

In an engine:

```python
from sparknet import rocenante
from sparknet.policy import TP4_POLICY

runtime = rocenante.AllReduce.from_exchange_group(
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
docs/                design, topology, nccl, rocenante, policy, roadmap, provenance
tests/               unit tests, the C proxy simulator, tests/gpu (torchrun)
benchmarks/          RoCEnante versus NCCL latency with a receipt
upstreams.lock.json  pinned revisions, patch heads and tree hashes
```

Tests: `make test` (stdlib unittest; the simulator needs a C compiler).
Everything in `docs/` states what was measured, where, and what remains
unqualified.

## License

Apache-2.0. Third-party attribution is in `NOTICE`.
