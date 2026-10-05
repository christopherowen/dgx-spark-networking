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
| `SPARKNET_ROCE_PROXY_CPU` | proxy thread placement: unset or `none` leaves it to the scheduler; a CPU number pins it to that core; `big` confines it to the big-core cluster, the cores above the midpoint between the smallest and largest `cpu_capacity` (GB10: the ten Cortex-X925 at 997 to 1024, excluding the ten Cortex-A725 at 718 to 731). `stats()` reports `proxy_cpus` and `proxy_cpu_observed`. Measured on the probe (see below), a candidate for the recipe environment pending a serving benchmark |

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
Proxy pinning is therefore a measured candidate for the recipe environment,
pending a serving benchmark (the serving process shares those cores). Those
runs pinned the single top core (cpu19); `big` now confines the thread to the
whole big-core set, so the scheduler can still move it off a busy core.

### Serving benchmark of the two families (2026-10-05)

spark-ds41f lab run `sparknet1` on the production TP4 recipe (r6 image, this
checkout mounted over its package; lean profile, one boot per arm, bracket
control at the end; `evidence/2026-10-05-tilelang-port/serving-bench`):

| Arm | Prose, one stream | JSON, one stream | Eight streams | Cold prefill 16K |
| --- | ---: | ---: | ---: | ---: |
| CuTe (control) | 62.46 tok/s, 31.81 ms/step | 99.83 tok/s, 37.76 ms/step | 173.9 tok/s | 5,667 tok/s |
| TileLang | 62.09 (-0.6%), 32.01 ms (+0.6%) | 98.85 (-1.0%), 38.15 ms (+1.0%) | 176.6 (+1.6%) | 5,676 |
| CuTe, proxy on big cores | 62.45 (-0.0%), 31.81 ms | 99.39 (-0.4%), 37.84 ms (+0.2%) | 178.3 (+2.6%) | 5,704 |
| TileLang, proxy on big cores | 61.78 (-1.1%), 32.15 ms (+1.0%) | 98.64 (-1.2%), 38.22 ms (+1.2%) | 175.2 (+0.8%) | 5,695 |
| CuTe again (bracket) | 62.40 (-0.1%), 31.85 ms (+0.1%) | 99.36 (-0.5%), 37.85 ms (+0.2%) | 174.7 (+0.4%) | 5,714 |

Both TileLang arms are 0.6 to 1.2 percent slower per single-stream decode
step, beyond the bracket's 0.1 to 0.2 percent drift; eight streams, prefill,
time to first token and mixed traffic are within noise, and the outputs are
identical. Proxy pinning has no measurable serving effect in this profile.
Neither is promoted: the rule is a measurable improvement in serving, and
the default stays `cute`. The step difference is not explained by the
collectives in isolation: `benchmarks/benchmark_oneshot.py` on the pair puts
the graph-replayed latencies equal (TileLang 1 percent faster at 768 KiB and
1 MiB) and TileLang's eager launch 15 us cheaper per call (27 against 43 us
at 8 KiB, 80 against 97 us for a 6 by 38,720 gather), so the cause is in the
serving context and still open.

### Where the NCCL cut is (2026-10-05)

One-shot forced up to the registered capacity against the NCCL-only profile
of the same fabric, BF16, two runs each (`evidence/2026-10-05-tilelang-port/cut-*`):

| Ring (TP4), all-reduce | One-shot (relay) | NCCL-only control |
| --- | ---: | ---: |
| 480 KiB | 94 to 96 us | 173 to 177 us |
| 640 KiB | 119 to 120 us | 165 to 195 us |
| 960 KiB | 158 to 162 us | 192 to 209 us |
| 1.25 MiB | 211 to 219 us | 215 to 223 us |
| 1.5 MiB | 252 to 259 us | 236 to 240 us |
| 2 MiB | 307 to 324 us | 270 to 282 us |

The ring's all-reduce crossover lies between 1.25 and 1.5 MiB against this
control, and the policy's own large-collective path is the balanced
four-channel NCCL, which is faster still, so the 1 MiB `tp4-ring` dispatch
holds and is slightly conservative; TP4's largest decode shape (96 tokens,
960 KiB) stays one-shot. The ring's one-shot all-gather beats the unbalanced
control through 4 MiB (710 against 914 us) but not the balanced NCCL the
profile uses (618 us at 4 MiB in `docs/nccl.md`), which puts that crossover
near 2 MiB, where the shard limit is.

| Pair, one cable | One-shot all-reduce | NCCL | One-shot all-gather | NCCL |
| --- | ---: | ---: | ---: | ---: |
| 480 KiB | 42 to 43 us | 114 to 115 us | 46 to 47 us | 111 to 115 us |
| 2 MiB | 136 to 137 us | 193 to 199 us | 147 to 148 us | 225 to 233 us |
| 4 MiB | 254 to 255 us | 318 to 329 us | 284 to 286 us | 407 to 409 us |

On a pair the one-shot wins at every size through 4 MiB by a roughly
constant 60 to 120 us, so the `tp2-direct` cut (dispatch equal to the 2 MiB
capacity) is too low. Raising it means raising the registered capacity,
which the vLLM fork also reads as its sequence-parallel threshold, so it is
a serving change to be measured, not a transport setting to flip. The
triangle (TP3) was not cabled and is unmeasured.

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
