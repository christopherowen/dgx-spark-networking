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

Streaming the relay in chunks and a shared progress window were measured
slower than whole-fragment forwarding (`2026-10-03-relay-progress`) and are
not carried. NIC forwarding (`mesh4`) is carried but not recommended.

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
