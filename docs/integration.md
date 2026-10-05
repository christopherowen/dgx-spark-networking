# Integration guide for recipe builders

This guide takes a serving recipe from a bare DGX Spark fabric to a running
engine that uses the library's collectives and NCCL settings. It assumes you
own the recipe (vLLM, SGLang or your own engine) and the cluster, and that
you will measure before you promote. The library describes, validates,
renders, probes and executes; it does not configure hosts, supervise serving
or judge model quality.

## 1. What a recipe takes from the library

| Need | Component | Form it reaches the recipe in |
| --- | --- | --- |
| A description of the fabric | `sparknet.topology` | `nodes.json`, validated against the chosen transport |
| The per-node settings | `sparknet.topology.render`, `sparknet.nccl.profiles` | environment variables, rendered once per node |
| The bulk collective library | `sparknet/nccl/patches`, `scripts/build-nccl.sh` | a rebuilt `libnccl.so.2` in the serving image |
| The small collectives | `sparknet.oneshot` | a runtime object the engine constructs per tensor-parallel group |
| The decision of who carries what | `sparknet.policy` | a `CollectivePolicy`, read from the same environment |
| Proof the fabric works | `sparknet.probe` | a bounded container per node, run before the model |
| An engine adapter | `sparknet.integration.vllm` | a drop-in class for the vLLM fork; the runtime API for everything else |

## 2. From a bare fabric to a rendered environment

### 2.1 Install

On the build host and in the serving image:

```sh
pip install git+https://github.com/christopherowen/dgx-spark-networking@v0.3.0                 # CPU tooling: topology, nccl, policy, probe planning
pip install 'dgx-spark-networking[runtime] @ git+https://github.com/christopherowen/dgx-spark-networking@v0.3.0'  # on the nodes: torch, cuda-python, nvidia-cutlass-dsl 4.7.1
```

The package itself has no dependencies, and the NCCL patch series ships in
it (`sparknet nccl patches --export DIR`), so an image build needs no
checkout of this repository.

The `runtime` extra pins the CuTe DSL to the version the vLLM nightly base
image installs; keep it at whatever your base image ships (the kernels need
4.7.x) and install the package with `--no-deps` inside such an image.

### 2.2 Describe the fabric

Start from the example that matches your cabling and edit the names,
management addresses, interface names and subnets:

```sh
sparknet topology example tp2-direct   > nodes.json   # two Sparks, one cable (a second cable is NCCL's)
sparknet topology example tp3-triangle > nodes.json   # three Sparks, every pair cabled
sparknet topology example tp4-ring     > nodes.json   # four Sparks in a loop
sparknet topology example switched     > nodes.json   # two to sixteen Sparks behind a switch
```

Or let the library read the fabric. For cabled fabrics it reads LLDP over
SSH (read-only) and writes the map and each node's netplan file under the
`10.<a><b>.<path>.<N>` rule; for switched fabrics it reads each host's active
rails from sysfs and derives the rail subnets from their live addresses:

```sh
sparknet topology discover dgx1 dgx2 dgx3 dgx4 --ssh-user spark --out site \
  --management-ip dgx1=192.0.2.1 --management-ip dgx2=192.0.2.2 \
  --management-ip dgx3=192.0.2.3 --management-ip dgx4=192.0.2.4 --management-interface enP7s7
sparknet topology discover dgx1 dgx2 dgx3 --fabric switched --ssh-user spark --out site \
  --management-ip dgx1=192.0.2.1 --management-ip dgx2=192.0.2.2 --management-ip dgx3=192.0.2.3 --traffic-class 106
```

Keep the site map out of git (`nodes.json` and `*.local.json` are ignored);
the examples carry documentation addresses only.

### 2.3 Validate

```sh
sparknet topology validate nodes.json --transport oneshot-ring4
```

Validation refuses rank order that does not follow the cables, routes that
are not reciprocal, a local interface used twice, unequal stripe counts, a
subnet that does not sit at exactly two endpoints, a four-node map without a
ring transport, and a switched map whose rails differ between nodes. It does
not prove a cable delivers packets; section 4 does.

### 2.4 Choose the transport and the profile

| Fabric | Transport | Profile | Qualification |
| --- | --- | --- | --- |
| 2 nodes, one cable | `oneshot-direct` | `tp2-direct` | triangle settings applied to one cable; not measured on the authors' fleet |
| 2 or 3 nodes, NCCL only | `nccl-direct` | `direct-nccl-only` | control |
| 3 nodes, triangle | `oneshot-direct` | `tp3-triangle` | promoted spark-ds41f baseline |
| 4 nodes, loop | `oneshot-ring4` | `tp4-ring` | measured balanced candidate |
| 4 nodes, loop, NCCL only | `nccl-ring` | `tp4-ring-nccl-only` | measured control |
| 4 nodes, loop, NIC forwarding | `oneshot-mesh4` | none | carried, not recommended |
| 2 to 16 nodes, switch | `oneshot-switched` | `switched` | clique mode, measured upstream on four switched Sparks; not on this fleet |
| 2 to 16 nodes, switch, NCCL only | `nccl-switched` | `switched-nccl-only` | control |

`sparknet nccl profiles` prints the same list with each profile's evidence
and the NCCL patches it needs. A profile is a measured set; treat an
unmeasured one as a starting point you owe a probe and a benchmark.

### 2.5 Render the environment

```sh
sparknet topology render nodes.json dgx3 --transport oneshot-ring4 --profile tp4-ring          # KEY=value lines
sparknet topology render nodes.json dgx3 --transport oneshot-ring4 --profile tp4-ring --json   # for a config file
```

The output is the complete fabric-related environment of that rank: the
profile's NCCL and one-shot settings, the ring policy for neighbour rings,
`NCCL_IB_HCA` with exact device names, the peer map or the rail list, the GID
index, the traffic class and the socket interfaces. Put it in the container's
environment verbatim. Every setting has exactly one name, `SPARKNET_ROCE_*`
(or NCCL's own `NCCL_*`).

Rendering refuses a profile that does not fit the map's transport or node
count, and an environment that contradicts the fabric policy (for example a
neighbour ring without `NCCL_ALGO=Ring`, or an NCCL-only transport with a
one-shot route set).

### 2.6 Hosts

`docs/host-prerequisites.md` lists what a node needs (kernel `kho=off`,
driver, firmware, rdma-core, MTU 9000, static addresses per stripe, GID 3 as
the IPv4 address, unlimited memlock). On each node, before every launch:

```sh
sparknet probe doctor nodes.json --node dgx3 --transport oneshot-ring4
```

It checks the map against the live devices, the GID in each declared subnet,
MTU, `kho=off`, memlock, the compiler and headers the proxy needs. It never
changes anything.

## 3. Build the serving image

Three things go into the image beyond the engine.

**The patched NCCL.** Every profile needs patch 0001 (the AArch64 send-path
fence; without it NCCL can hang every rank); the `tp4-ring` profile needs
0002 to 0004. Build once and replace the wheel's library:

```sh
scripts/build-nccl.sh /opt/nccl 8        # from a checkout
# or, from the installed package, in an image build:
sparknet nccl patches --export /workspace/nccl-patches
git clone --filter=blob:none https://github.com/NVIDIA/nccl.git /workspace/nccl
git -C /workspace/nccl checkout --detach 73cf112295c33aee2b895f329f592f2a9b4b0f97   # v2.30.7-1, upstreams.lock.json
for p in $(grep -v '^#' /workspace/nccl-patches/series); do git -C /workspace/nccl am /workspace/nccl-patches/$p; done
make -C /workspace/nccl -j8 src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
# in the image: replace the nvidia-nccl wheel's libnccl.so.2 and keep the hash beside it
```

The spark-ds41f Dockerfile checks at build time that `ncclGetVersion` reports
2.30.7 and that the installed file's SHA-256 is the built one; do the same.

**The one-shot proxy.** It is plain C over libibverbs, built with the host
compiler on first use into `SPARKNET_ROCE_CACHE_DIR`. Build it at image time
so no compiler runs at startup:

```sh
apt-get install -y libibverbs-dev gcc
SPARKNET_ROCE_CACHE_DIR=/opt/sparknet/roce python3 -c \
  'from sparknet.oneshot._proxy import load; print("proxy ABI", load().roce_abi_version())'
```

**The kernels.** Both families compile on the first `prepare` on a GPU and
cache on disk through their toolchains' own variables: the CuTe DSL under
`CUTE_DSL_CACHE_DIR`, TileLang under `TILELANG_CACHE_DIR` (default
`~/.tilelang/cache`). Those caches are shared with every other kernel of
the same toolchain in the serving process, so the recipe, not this library,
places them with its other JIT caches on a mounted volume (spark-ds41f uses
`/cache/kkref/jit/<toolchain>`). Either warm that cache on each
node once (a `prepare` for every dtype the engine reduces, which the probe
does) and mount it into the serving container, or run the warm-up during the
image build on a Spark with `--gpus=all`. After the engine's own warm-up,
call `sparknet.oneshot.freeze_kernel_resolution()`: a compile inside a
serving step then raises instead of stalling the step.

The serving container needs the RDMA devices and locked memory, the same
flags the probe container uses plus shared memory for the engine:

```text
--network=host --ipc=host --gpus=all
--device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK
--ulimit=memlock=-1:-1 --ulimit=stack=67108864:67108864 --shm-size=16g
```

[`docker/Dockerfile`](../docker/Dockerfile) is the reference for exactly
this: an `nccl-builder` stage that exports the packaged series, applies it to
the pinned release, checks the tree hash and runs `make src.build` for
`sm_121`; a runtime stage that installs the rebuilt library over the wheel's,
installs the package with `--no-deps`, builds the proxy into
`SPARKNET_ROCE_CACHE_DIR`, and ends with an import check that fails the
build if the proxy ABI or the NCCL hash is not what was built. The engine
stages (spark-ds41f's vLLM fork, B12X, TileLang) are the recipe's own; CI
lints the file and the image is built by hand on a Spark.

## 4. Qualify the fabric before the model

Render one bounded probe command per node and run them together, with
serving stopped, inside whatever cluster window your site uses:

```sh
sparknet probe render-command nodes.json dgx1 --transport oneshot-ring4 --profile tp4-ring \
  --image your-image:tag --port 29650 -- --benchmark --counter-samples --port-samples
```

The command runs `sparknet/probe/collectives.py` in a separate IPC
namespace with a 12 GiB limit and a 600 s timeout. It constructs the runtime
from the rendered environment, checks that the runtime applied the
requested topology and dispatch limit, then checks exact BF16 and FP32
all-reduce, all-gather and reduce-scatter results for tiny, unaligned and
large messages and both sides of every dispatch boundary, in eager execution
and across four CUDA graph replays with changing inputs. With `--benchmark`
it screens steady graph latency per operation and size, with RDMA error and
physical-port counters around every timed case, and asserts that the proxy's
byte counters agree with the policy for each case. Every rank must print a
passing JSON result and exit zero. Keep the NCCL logs; check the ring order
and the IB transport on each node.

For a new transport or a changed kernel, also run the GPU test
(`tests/gpu/test_oneshot_gpu.py`, torchrun, one process per node): NCCL
parity, bit-identical ranks, graph replay, alternating streams, a proxy that
misses a doorbell, and fault injection that proves every rank fails stop.

`sparknet probe fleet` runs that command on every node at once over ssh
(one session per node, the ranks rendezvous on the head's management
address), keeps `<node>.log` and `<node>.json` under `--out`, and prints the
latency table; `--dry-run` shows the ssh commands first. `--package-source
DIR` mounts a checkout's `sparknet/` over the image's installed package, so a
change can be probed through a released image, and `--env KEY=VALUE` adds a
candidate setting for an A/B. `sparknet probe summarize control=DIR1
candidate=DIR2` tabulates two sets of receipts with deltas. To measure a
pair out of a ring or a triangle without recabling,
`sparknet topology subset nodes.json dgx2 dgx3 --out pair.json` carves the
two-node map (a `tp2-direct` fabric) from the site map.

A passing probe is a prerequisite for serving, not serving acceptance.

## 5. Wire the engine

### 5.1 The runtime contract

Whatever the engine, these hold:

- **Construct collectively, over a CPU group.** Every rank of the
  tensor-parallel group calls `AllReduce.from_exchange_group` together with
  identical limits. The exchange group is a gloo group: a torch NCCL group
  would create a torch NCCL communicator the engine never uses, costing about
  3.4 GB of unified memory per rank.
- **Prepare before capture.** `prepare(dtypes, padded_gather=True)` compiles
  the launchers for the dtypes the engine reduces and allocates scratch. A
  dtype first used inside a CUDA graph capture raises.
- **Dispatch is rank-invariant.** Decide with `CollectivePolicy` (or the
  runtime's `should_allreduce` and `should_all_gather`) from dtype, shape,
  contiguity and byte size only. Never from pointers, never per rank.
- **One stream inside a capture.** Eager collectives on different streams are
  ordered by the runtime; inside `torch.cuda.graph` every collective must be
  on the capture stream, under `runtime.capture(stream=...)`.
- **Check health after each step's own host sync.** `check_health()` costs two
  host reads. A wait that timed out poisons the runtime: later launches do
  nothing, `check_health` raises, and the peers time out and raise too. The
  engine treats this as a fatal error of the rank group. There is no
  fallback and no reconnect; restart the group.
- **Choose the spin limit as the failure-detection latency.** About a
  microsecond per poll; the profiles use 5,000,000 (seconds). It must exceed
  the longest legitimate rank skew, which for tensor-parallel decode is
  milliseconds.
- **Close on shutdown.** `close()` synchronizes the device and releases the
  queue pairs; peers that still write see a completion error and raise.

Memory: the pinned region is `(2 x world_size + 2)` slots of
`max(capacity, gather limit)` bytes plus flags, about 20 MiB for four ranks
at 2 MiB; `prepare` adds two capacity-sized alignment buffers and, with
`padded_gather`, `(world_size + 1)` gather-limit buffers on the device.

### 5.2 vLLM (Local Inference Lab fork)

`sparknet.integration.vllm.SparknetOneShotAllReduce` replaces the fork's
`B12xRoceAllReduce` with the same constructor keywords and methods. The
one-file switch and the policy guards it keeps are described in
`sparknet/integration/vllm/README.md`; spark-ds41f carries it as its vLLM
patch 0038 and serves both promoted recipes through it.

Serving arguments per transport:

| Transport | Serve arguments |
| --- | --- |
| `oneshot-*` | custom all-reduce enabled (do not pass `--disable-custom-all-reduce`); the engine constructs the one-shot adapter, which is always on, and reads its limits (`SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES`, `SPARKNET_ROCE_ALLGATHER_MAX_BYTES`) from the rendered environment |
| `nccl-*` | `--disable-custom-all-reduce`; the engine constructs no one-shot adapter |
| all | `--distributed-executor-backend mp --nnodes N --tensor-parallel-size N`; one TP group in cable order; no expert, pipeline, data or context parallel groups on a neighbour ring |

Per node, the same command plus `--node-rank <rank> --master-addr <head
management ip> --master-port <port>`, with `--headless` on every node but the
head, and `VLLM_HOST_IP=<that node's management ip>`. Full CUDA graphs for
decode are supported: the adapter prepares at construction, before capture.

### 5.3 A custom engine

`examples/minimal_engine.py` is the complete pattern in ninety lines. The
core:

```python
from sparknet import oneshot
from sparknet.policy import CollectivePolicy

policy = CollectivePolicy.from_environment(dict(os.environ))
runtime = oneshot.AllReduce.from_exchange_group(
    exchange_group=cpu_group, device=device,
    max_size=policy.all_reduce_capacity_bytes, max_gather_bytes=policy.all_gather_shard_bytes)
runtime.prepare((torch.bfloat16, torch.float32), padded_gather=True)

def all_reduce(x):
    if policy.all_reduce_backend(x.numel() * x.element_size(), dtype_name(x), contiguous=x.is_contiguous()) == "oneshot":
        return runtime.all_reduce(x)
    out = x.clone(); dist.all_reduce(out); return out

def all_gather(x, dim):
    if policy.all_gather_backend(x.numel() * x.element_size(), dtype_name(x), dim=dim, ndim=x.dim()) == "oneshot":
        return runtime.all_gather(x, dim=dim)
    parts = [torch.empty_like(x) for _ in range(world)]; dist.all_gather(parts, x); return torch.cat(parts, dim=dim)
```

Reduce-scatter and every variable collective stay on NCCL. After the step's
`synchronize` (or the host copy of the sampled tokens), call
`runtime.check_health()`.

### 5.4 SGLang and other engines

No adapter ships for SGLang yet. The integration point is the same place
its custom all-reduce lives (the device communicator of its distributed
layer): construct the runtime per TP group over a gloo group, prepare,
dispatch by the policy, capture on one stream, check health per step. The
fabric work (map, profile, image, probe) is engine-independent and should
be done first; the probe validates the fabric without any engine.

## 6. Policy and limits

| Limit | `tp3-triangle`, `tp2-direct`, `switched` | `tp4-ring` | Why |
| --- | --- | --- | --- |
| All-reduce dispatch | 2 MiB | 1 MiB | measured crossover with NCCL Ring on four nodes is near 1 to 1.25 MiB |
| All-reduce capacity | 2 MiB | 2 MiB | registered and primed; vLLM's sequence-parallel prefill floor reads it |
| All-gather input shard | 4 MiB | 2 MiB | NCCL wins above it on the ring |

To tune: copy the profile into your recipe's own catalog, change one
variable, render, probe with `--benchmark`, then measure serving. Dispatch
and capacity are deliberately separate; lowering capacity also moves vLLM's
sequence-parallel threshold, which is a model-scheduling change.

## 7. Operate

- The proxy thread polls a doorbell on the host, and on the GB10 the scheduler
  may place it on a little core: the probe measured a 16 to 20 us spread at
  10 KiB unpinned against 17.8 to 18.4 us with `SPARKNET_ROCE_PROXY_CPU=big`
  (`docs/oneshot.md`). Put that variable in the recipe's container environment
  to confine the proxy to the big cores; promote it on a serving benchmark,
  since the serving process shares those cores. `runtime.stats()` shows
  `proxy_cpus` and `proxy_cpu_observed`.
- Log `runtime.stats()` at startup: topology, HCAs, stripe count, dispatch
  limit, and later the per-HCA byte counters. On a ring the bytes must be
  equal across interfaces.
- Run `sparknet probe doctor` in the launch preflight; a changed GID index
  after a reboot or a moved cable shows up there, not in a hang.
- A poisoned runtime (timeout, proxy failure) is a rank-group failure.
  Restart the group; do not restart one rank.
- Keep the rendered environment with the serving configuration and treat
  it as part of the image's identity; the probe's JSON result and the NCCL
  init logs are the fabric receipt for that configuration.

## 8. Before promotion

1. `sparknet topology validate` and `sparknet nccl validate` pass for the
   map and profile.
2. `sparknet probe doctor` passes on every node after a reboot.
3. The collective probe passes on every rank, with counters, twice, including
   once after a cold restart; the ring order and the IB transport are in the
   logs; the proxy byte counters match the policy.
4. The engine boots with full CUDA graphs and `check_health` wired, serves
   the recipe's quality gate, and its decode and prefill numbers are recorded
   against a control with the same image and a pinned verification setting.
5. Sustained load with temperatures and memory recorded; the fabric does
   not change them, but a balanced ring moves heat around.

## 9. Troubleshooting

| Symptom | Likely cause | Where to look |
| --- | --- | --- |
| `RDMA device X not found` at construction | device names in the peer map or `NCCL_IB_HCA` do not match `ibv_devices` | `sparknet topology inventory` on the node |
| `port 1 is not active` | cable or link down, or the wrong port cabled | `rdma link`, `sparknet probe doctor` |
| `ibv_reg_mr(pinned region): Cannot allocate memory` | `kho=off` missing from the kernel command line, or memlock limited | `cat /proc/cmdline`, `ulimit -l`, container `--ulimit memlock=-1:-1` |
| `rank N published incompatible transport geometry` | ranks run different topologies, stripe counts or ABIs (mixed images or environments) | compare each node's rendered environment and image |
| `timed out waiting for rank N, HCA h` | that peer stalled, its GID is wrong for the subnet, or the spin limit is below the real skew | `sparknet probe doctor` on rank N; NCCL/one-shot GID index; spin limit |
| `one-shot policy unavailable: needs an integrated GPU with an active RDMA device` | not a Spark, or no active device at the GID index | `sparknet topology inventory` |
| `SWITCHLESS_BIDIRECTIONAL needs ...` in the NCCL log | channel count not even or not a multiple of four for mode 2 | `sparknet nccl validate` |
| NCCL hangs with every rank waiting, AArch64 | unpatched NCCL (missing the CTS fence) | image's `libnccl.so.2` hash against the built one |
| the probe fails `actual transport counters disagree with the dispatch policy` | the engine or the probe routed a size to the wrong backend; environment and runtime limits differ | the rendered environment versus `runtime.stats()` |
| retransmissions in `roce_adp_retrans` during the probe | lossy fabric: a switch without PFC/ECN, or NIC forwarding (`oneshot-mesh4`) | `roce_traffic_class`, switch configuration, use the host relay |

## 10. Reference

**CLI**

```text
sparknet topology validate|render|example|subset|discover|inventory
sparknet nccl env|validate|profiles|patches [--export DIR]
sparknet policy show
sparknet probe doctor|gpudirect|render-command|fleet|summarize|collectives
```

**Profile-level environment** (from `sparknet nccl env`): `NCCL_*` as listed
in `docs/nccl.md`; `SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES`,
`SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES`,
`SPARKNET_ROCE_ALLGATHER_MAX_BYTES`, `SPARKNET_ROCE_SPIN_LIMIT`; per host,
`SPARKNET_ROCE_PROXY_CPU` (`docs/oneshot.md`).

**Per-node environment** (from `sparknet topology render`):
`SPARKNET_ROCE_TOPOLOGY`, `SPARKNET_ROCE_PEER_HCAS` (cabled) or
`SPARKNET_ROCE_HCA` (switched), `SPARKNET_ROCE_GID_INDEX`,
`SPARKNET_ROCE_TRAFFIC_CLASS`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`,
`NCCL_IB_TC`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`,
and the ring policy keys on neighbour rings.

**Runtime API** (`sparknet.oneshot`): `AllReduce.from_exchange_group(exchange_group, device, max_size, max_gather_bytes)`,
`prepare(dtypes, padded_gather=)`, `should_allreduce(x)`, `all_reduce(x, out=, stream=)`,
`should_all_gather(x, dim)`, `all_gather(x, dim=, out=, stream=)`, `capture(stream=)`,
`check_health()`, `poisoned`, `stats()`, `close()`; properties `topology`,
`hca_names`, `stripe_count`, `dispatch_max_bytes`, `max_size`, `max_gather_bytes`;
module functions `is_supported(device)`, `discover_hcas()`, `default_gid_index()`,
`freeze_kernel_resolution(reason)`.

**Policy API** (`sparknet.policy`): `CollectivePolicy(dispatch, capacity, gather_shard)`,
`.from_environment(env)`, `.environment()`, `.all_reduce_backend(nbytes, dtype, contiguous=)`,
`.all_gather_backend(shard_bytes, dtype, dim=, ndim=, contiguous=)`, `.reduce_scatter_backend()`,
`policy_for_profile(name)`, `TP3_POLICY`, `TP4_POLICY`.
