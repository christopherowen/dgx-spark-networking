# One-shot collectives

`sparknet.oneshot.AllReduce` is the one-shot RDMA all-reduce and
all-gather runtime (b12x calls the module RoCEnante, a pun on RoCE and
Rocinante; the mechanism's name is the one-shot collective) for tensor parallelism across DGX Spark nodes joined by
their ConnectX-7 ports. The protocol is described in `docs/design.md`; this
page is the interface, the contract, the environment and the evidence.

## Interface

```python
rt = AllReduce.from_exchange_group(exchange_group=cpu_group, device=device,
                                   max_size=2 << 20, max_gather_bytes=2 << 20)
rt.prepare((torch.bfloat16, torch.float32, torch.float16), padded_gather=True)
rt.should_allreduce(x)        # dtype, shape, contiguity, byte size <= dispatch limit
y = rt.all_reduce(x)          # out=, stream= optional
rt.should_all_gather(x, dim)  # dim 0 or the last dim, shard <= max_gather_bytes
g = rt.all_gather(x, dim=-1)
with rt.capture(stream=s): ...  # inside torch.cuda.graph; one stream per capture
rt.check_health()             # after each step's own device-to-host sync
rt.stats(); rt.poisoned; rt.close()
```

Constructor keywords: `topology` (`direct`, `ring4`, `mesh4`; default from
the environment), `hca_names`, `peer_hca_names`, `gid_index`, `threads`,
`blocks`, `transport` (the posting implementation, default the host proxy).
Exchange setup over a CPU (gloo) group: a torch NCCL group would create a
torch NCCL communicator costing about 3.4 GB of unified memory per rank.

Constraints: 2 to 16 ranks (`ring4`/`mesh4` need exactly 4), one collective
in flight per runtime, integrated GPU with unified addressing, active RDMA
devices, messages a multiple of 16 bytes.

## Contract

**Eligibility is rank-invariant.** `should_allreduce` and `should_all_gather`
decide from dtype, shape, contiguity and byte size only. An unaligned pointer
is staged through scratch, never declined. A closed runtime raises instead
of declining, so no rank can fall back alone. `should_allreduce` applies the
dispatch limit; `all_reduce` executes anything up to the registered capacity
when called explicitly.

**Configuration is checked at setup.** Every rank publishes API version,
proxy ABI, topology, lane count, slot geometry, limits, dispatch limit, spin
limit, traffic class and launch geometry; any difference fails construction
on every rank. The proxy's connect validates every rank's record, including
the opposite rank in `ring4`, before connecting anything.

**Failures are fail-stop, never a fallback.** A wait beyond the spin limit
records the sequence and the missing peer, the kernel skips its data phase
and keeps the epoch, every later launch is a no-op, and `check_health`
raises. Peers starve on the stalled rank and poison themselves within one
spin limit. Choose the spin limit as the failure-detection latency (about a
microsecond per poll; the profiles use 5,000,000).

**Streams.** Collectives on different streams are ordered with an event;
inside a capture every collective uses the capture stream.

**Memory.** The pinned region is `world_size x 2` receive slots plus 2 send
slots of `max(max_size, max_gather_bytes)` bytes, plus flags. `prepare`
allocates two `max_size` alignment buffers and, with `padded_gather`, the
padded-gather scratch.

**No JIT in a step.** `prepare` compiles the launchers in process; a dtype
not prepared before a capture raises. Serving images warm the CuTe DSL cache
at build time and call `freeze_kernel_resolution()` after warm-up.

## Environment

Each setting has one `SPARKNET_ROCE_*` name; the HCA list, GID index and
traffic class fall back to NCCL's own settings.

| Name | Meaning |
| --- | --- |
| `SPARKNET_ROCE_TOPOLOGY` | `direct`, `ring4` or `mesh4` |
| `SPARKNET_ROCE_PEER_HCAS` | JSON `{peer rank: [local HCAs in stripe order]}`; neighbours only for ring4 |
| `SPARKNET_ROCE_HCA` | explicit HCA list for direct cliques (falls back to `NCCL_IB_HCA`) |
| `SPARKNET_ROCE_GID_INDEX` | IPv4 RoCE v2 GID slot (falls back to `NCCL_IB_GID_INDEX`, default 3) |
| `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES` | dispatch ceiling, at most the registered capacity |
| `SPARKNET_ROCE_SPIN_LIMIT` | polls before a wait times out |
| `SPARKNET_ROCE_TRAFFIC_CLASS` | DSCP/ECN byte for every QP (falls back to `NCCL_IB_TC`, default 0) |
| `SPARKNET_ROCE_CACHE_DIR` | where the proxy `.so` is built (default `<XDG cache>/sparknet/roce`) |
| `SPARKNET_ROCE_MESH_ROTATE` | mesh4 only: rotate posting order (measured no benefit; keep 0) |
| `SPARKNET_ROCE_KERNELS` | kernel family, `cute` (default) or `tilelang`; see Kernel families below |
| `SPARKNET_ROCE_PROXY_CPU` | proxy thread placement: unset or `none` leaves it to the scheduler; a CPU number pins it; `big` pins it to the CPU with the highest `cpu_capacity` (GB10: ten Cortex-X925 and ten Cortex-A725). Candidate, unmeasured: the probe's `proxy_cpu_observed` shows where the thread ran |

The capacity and all-gather limits are constructor arguments; the vLLM
adapter reads `SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES` and
`SPARKNET_ROCE_ALLGATHER_MAX_BYTES`. One-shot is always on where it is
constructed; there is no switch.

## Evidence

Measured in spark-ds41f on the four Sparks (GB10, driver 580.178.04,
kernel 7.0.0-1019-nvidia-64k), BF16 all-reduce, median of the slowest rank
per sample of 256 graph-replayed calls:

| Input per rank | NCCL ring (1 ch) | Host relay, clockwise | Bidirectional relay |
| --- | ---: | ---: | ---: |
| 10 KiB | 90.6 to 92.4 us | 17.4 to 18.2 us | 16.9 us |
| 60 KiB | 111.5 to 114.2 us | 28.1 to 29.6 us | 28.0 us |
| 480 KiB | 168.9 to 174.1 us | 95.1 to 96.3 us | 94.0 us |
| 2 MiB | 359.5 to 362.1 us | 314.8 to 316.3 us | 307.7 us |

The bidirectional relay changed each node's proxy bytes from about
32.04/16.02 GB clockwise/counterclockwise to 24.03/24.03 GB. Three
control/candidate/control launches passed exact-data checks, changing
CUDA-graph inputs, dispatch boundaries and zero tracked RDMA errors
(`2026-10-03-ring4-bidirectional`). The three-node direct mode is the
promoted spark-ds41f TP3 baseline; on four nodes with the relay and the tuned NCCL
profile, decode throughput was 14 to 27 percent above the three-node
baseline in a matched comparison (`2026-10-03-tp3-tp4-comparison`), a
historical control rather than a fresh pair.

### Measured through this package (2026-10-05)

First run of the package's own GPU suite and probe on the fleet, r6 image,
both kernel families (`evidence/2026-10-05-tilelang-port`). BF16, median
over runs of the slowest-rank median; the pair is `dgx1`-`dgx2` carved from
the four-node ring (one cable, two PCIe-path stripes, `tp2-direct`), the
ring is `tp4-ring`. "NCCL" is the NCCL-only control profile on the same
fabric and image.

| Pair of Sparks, one cable | One-shot (CuTe, 3 runs) | One-shot (TileLang, 3 runs) | NCCL only (2 runs) |
| --- | ---: | ---: | ---: |
| all-reduce 10 KiB | 10.9 us | 10.9 us | 67.2 to 77.3 us |
| all-reduce 60 KiB | 17.0 us | 16.9 us | 80.0 to 87.8 us |
| all-reduce 480 KiB | 43.1 us | 42.6 us | 109.1 to 122.6 us |
| all-reduce 2 MiB | 137.1 us | 135.5 us | 191.7 to 205.2 us |
| all-gather 10 KiB | 10.9 us | 10.9 us | 65.2 to 72.7 us |
| all-gather 60 KiB | 17.1 us | 16.8 us | 81.7 to 92.8 us |
| all-gather 480 KiB | 45.5 us | 46.0 us | 112.1 us |
| all-gather 2 MiB | 147.9 us | 147.1 us | 223.8 to 233.1 us |

| Four-node ring | One-shot (CuTe, 3 runs) | One-shot (TileLang, 3 runs) | NCCL only (1 run) |
| --- | ---: | ---: | ---: |
| all-reduce 10 KiB | 16.2 us | 18.1 us | 99.7 us |
| all-reduce 60 KiB | 30.4 us | 28.5 us | 128.3 us |
| all-reduce 480 KiB | 96.6 us | 92.9 us | 178.4 us |
| all-reduce 2 MiB (policy: NCCL) | 271.9 us | 263.6 us | 273.9 us |
| all-gather 10 KiB | 19.1 us | 21.5 us | 95.0 us |
| all-gather 60 KiB | 30.6 us | 30.0 us | 127.8 us |
| all-gather 480 KiB | 106.6 us | 103.2 us | 180.9 us |
| all-gather 2 MiB | 354.0 us | 353.7 us | 477.1 us |

The ring numbers reproduce the earlier spark-ds41f measurements above. The
two families are bit-identical (the GPU suite's `test_kernel_families_*`,
pair and ring); TileLang is 3 to 5 percent faster at 480 KiB in every ring
run and about 1 percent faster at 2 MiB on the pair, and the 10 KiB gap in
the full ring runs did not survive an interleaved series at 10 and 60 KiB
(twelve runs: both families spread over 17.8 to 20.4 us unpinned, with no
offset between them). In that series the proxy thread's placement explained
the spread: with `SPARKNET_ROCE_PROXY_CPU=big` every run landed in 17.8 to
18.4 us, and the one slow CuTe full run had its proxy on a little core.
Proxy pinning is therefore a measured candidate for the profiles, pending
a serving measurement (the serving process shares those cores).

Streaming the relay in chunks and a shared progress window were measured
slower than whole-fragment forwarding (`2026-10-03-relay-progress`) and are
not carried. NIC forwarding (`mesh4`) is carried but not recommended.

## Kernel families

Two kernel families carry the protocol over the same pinned region, proxy,
launcher signature and runtime. ``cute`` is the vendored CuTe DSL pair
(`_oneshot_cute.py`, `_allgather_cute.py`, intrinsics in
`_cute_intrinsics.py`); ``tilelang`` is the TileLang pair
(`_oneshot_tilelang.py`, `_allgather_tilelang.py`), generated as CUDA source
with the protocol steps in `_device.py`: the same PTX for every system-scope
load, store and fence, the same spin, the same float32 accumulation in
fixed rank order and the same conversions, so the two families are meant to
be bit-identical and the GPU test checks it (`test_kernel_families_*`).
`SPARKNET_ROCE_KERNELS` selects the family for a process, the runtime's
`kernels` keyword (and the probe's `--kernels`) for one runtime; every rank
must agree, which the setup handshake enforces. The TileLang family passed the GPU suite and the probe on the pair and the
ring on 2026-10-05 (see Measured through this package); the default stays
`cute` until the deployment repository has benchmarked serving with
`SPARKNET_ROCE_KERNELS=tilelang`, which is the promotion gate. The TileLang
family needs `tilelang` (the image's) and compiles with nvcc on first
`prepare`, in well under a second per launcher, into TileLang's own cache.

## Tests

- `tests/test_oneshot_cpu.py`: the production C proxy under a fake verbs
  layer with sanitizers (26,000 collectives), the topology resolver, the
  kernels' flag selection.
- `tests/gpu/test_oneshot_gpu.py` (torchrun, 2 or more nodes): NCCL
  parity, bit-identical ranks, eligibility and unaligned staging, dim-0,
  last-dim and padded gathers, graph replay mixing both collectives, the
  adapter call pattern, alternating streams, a proxy that misses a doorbell,
  fault injection in eager and graph mode.
- `benchmarks/benchmark_oneshot.py`: latency against NCCL with a receipt.
- `sparknet probe collectives`: the policy on the actual fabric.
