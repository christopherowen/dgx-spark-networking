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

# Fused halves (reference kernels: TileLang family):
rt.send(x); y = rt.receive()  # the all-reduce in two halves
args = rt.fused_send(nbytes, dtype)  # a producer kernel of yours runs the send half
args = rt.fused_receive()     # a consumer kernel of yours runs the receive half
src = rt.device_header()      # the CUDA functions those kernels call
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

## Fused halves

The all-reduce kernel stages its input into the pinned send slot, rings the
proxy's doorbell, waits for the peers' payloads and reduces them. The halves
let other kernels do the first two or the last two steps themselves: a
producer writes its output straight into the send slot and rings the
doorbell, a consumer waits and takes the reduced values into its own
registers. That removes the staging copy and a kernel boundary from the
critical path, which only a kernel generated with the protocol functions can
do. The result is bit-identical to the standalone all-reduce.

`device_header()` is CUDA source (TileLang: `T.import_source`; CUDA C++: a
header) with this rank's constants and the `roce_fused_*` functions; compile
it per rank. `FusedArgs` carries the counters tensor, the pinned region's
address, the payload size and the trace file's address.

Producer kernel (send half), launched right after `fused_send(nbytes, dtype)`
on the same stream:

```c
roce_u32 seq = roce_fused_seq(counters);
if (!roce_fused_poisoned(counters)) {
    // write the payload, 16-byte packs in the dtype's layout, to
    // roce_fused_send_slot(region_base, seq) + offset (or roce_fused_send_slot_at)
    __syncthreads();
    if (threadIdx.x == 0) roce_fused_send_commit(counters, gridDim.x, region_base, nbytes, seq);
}
```

Consumer kernel (receive half), launched right after `fused_receive()` on the
send's stream, with at least `world_size x lanes` threads per block:

```c
roce_u32 seq = roce_fused_seq(counters);
if (!roce_fused_poisoned(counters)) {
    roce_fused_wait(threadIdx.x, region_base, seq, counters);
    __syncthreads();
    if (!roce_fused_poisoned(counters)) {
        roce_pack p = roce_fused_sum_pack_bf16(region_base, seq, offset);  // or _f32, _f16
        // ... use p (the standalone kernel's output bits at offset) ...
    }
    roce_fence_sc_gpu();
    __syncthreads();
    if (threadIdx.x == 0) roce_fused_receive_tail(counters, gridDim.x, region_base, seq);
}
```

Rules: between the halves nothing else of this runtime runs (the runtime
raises on every rank before enqueueing anything); every block of a half calls
its commit or tail exactly once, and the halves' counters reset themselves,
so any grid size works; a poisoned runtime turns both halves into no-ops that
the host check reports. `send`/`receive` are the reference kernels for each
half (`_fused_tilelang.py`), and any pairing of them with your kernels holds.
`prepare(fused=True)` compiles them before a capture. A traced runtime renders the header with the
trace layout (`roce_trace_begin`, `roce_trace_stamp`), which the reference
kernels use to fill the same trace words as the standalone kernel.

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
| `SPARKNET_ROCE_TRACE` | `1`: per-op timing trace of the all-reduce (both families), see `sparknet.oneshot.trace` |
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
The collectives in isolation did not explain it: `benchmarks/benchmark_oneshot.py`
on the pair puts the graph-replayed latencies equal and TileLang's eager
launch 15 us cheaper per call.

**The cause: the TileLang kernel's register footprint.** Torch-profiler
traces of a 128-token decode on every rank (`evidence/2026-10-05-tilelang-port/decode-profiles`)
put the whole difference in the graph-replayed 4-block all-reduce, the
60 KiB decode shape. Almost every one of those launches runs beside the
model's L2 weight prefetch (vLLM `l2_prefetch.py`, which issues
`cp.async.bulk.prefetch.L2` fills on a side stream right before each
all-reduce), and there the TileLang kernel's floor was about 15 us higher on
every rank (5th percentile 40 against 27 us on the MoE all-reduce), while it
was faster than CuTe when no prefetch ran. The two kernels issue the same
memory-ordering instructions; the difference was that nvcc, compiling
TileLang's CUDA with `__launch_bounds__(512, 1)`, spent 54 to 56 registers
per thread (2 resident blocks per SM) where the CuTe kernel uses 40 (3 per
SM, the warp limit at 512 threads). With the kernels declaring full
residency (`T.annotate_min_blocks_per_sm`, 40 registers) the profile matches
CuTe on every rank:

| dgx1, decode profile | CuTe | TileLang, 56 registers | TileLang, 40 registers |
| --- | ---: | ---: | ---: |
| MoE all-reduce p5 / p50 | 26.6 / 44.8 us | 40.2 / 63.6 us | 26.2 / 45.6 us |
| Attention all-reduce p5 / p50 | 35.2 / 51.3 us | 48.2 / 59.2 us | 33.2 / 51.4 us |
| Window span | 1,562.8 ms | 1,575.9 ms | 1,560.2 ms |

The 40-register column was measured on a variant with the stage, reduce and
gather loops moved into the CUDA header. The kernels as committed keep those
loops in TileLang; profiled beside CuTe in one later window they match it as
well (dgx1: span 1,558.8 against 1,561.3 ms, MoE all-reduce p5 / p50 26.8 /
44.5 against 25.6 / 43.6 us, every rank within the run-to-run spread).

Ruled out on the way, each measured: code size (the header-loop variant cut
the kernel from 400 to 312 instructions and changed nothing, so it was not
kept), extra
memory traffic (identical fences, invalidates and strong loads in the two
SASS listings), runtime threads, programmatic dependent launch, cache
carveout and launch attributes (TVM launches with a plain
`cuLaunchKernel`). Microbenchmarks did not reproduce the effect; only the
serving profile did.

The serving benchmark with the 40-register header-loop variant
(`evidence/2026-10-05-tilelang-port/serving-bench-regs40`, same lean profile,
bracket) puts the two families level: single-stream step +0.1 percent (prose)
and +0.3 percent (JSON) against a bracket of 0.0 and +0.2 percent, eight
streams and prefill within noise. TileLang no longer loses, but it is not
measurably better, so the default stays `cute`.

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
